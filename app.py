import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import uuid
import json
import hashlib

# ============================
# PASSWORD (SET)
# ============================
PASSWORD_HASH = hashlib.sha256("enflowers".encode()).hexdigest()

def check_password():
    def password_entered():
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == PASSWORD_HASH:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔐 Enter password to access NHS Atlas", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔐 Enter password to access NHS Atlas", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False
    else:
        return True

# ============================
# APP CONFIG
# ============================
st.set_page_config(page_title="NHS Atlas ULTRA (Cloud)", layout="wide")

if not check_password():
    st.stop()

# ============================
# DEMO DATA + STATE
# ============================
ORG_LIST = ["TRUST_A", "TRUST_B", "ICB_DEMO"]

def _seed_data():
    now = datetime.utcnow()
    rng = np.random.default_rng(42)

    beds = []
    for org in ["TRUST_A", "TRUST_B"]:
        occ_rate = 0.92 if org == "TRUST_A" else 0.86
        for ward in ["AMU", "SAU", "Respiratory", "Surgery", "Neuro"]:
            for _ in range(30):
                beds.append({"org_id": org, "ward": ward, "is_occupied": bool(rng.random() < occ_rate)})
    beds = pd.DataFrame(beds)

    # ED arrivals hourly (14d)
    ed = []
    hours = 14 * 24
    for org in ["TRUST_A", "TRUST_B"]:
        base = 11 if org == "TRUST_A" else 9
        for h in range(hours):
            t = now - timedelta(hours=(hours - 1 - h))
            baseline = base + 3 * np.sin((h / 24.0) * 2 * np.pi)
            arrivals = max(0, int(rng.normal(baseline, 3)))
            ed.append({"org_id": org, "ts": t.replace(minute=0, second=0, microsecond=0), "value": arrivals})
    ed = pd.DataFrame(ed)

    # Inpatient stays (LOS)
    ip = []
    for org in ["TRUST_A", "TRUST_B"]:
        mean_los = 4.6 if org == "TRUST_A" else 4.0
        for _ in range(900):
            admit = now - timedelta(days=int(rng.integers(0, 45)), hours=int(rng.integers(0, 24)))
            if rng.random() < (0.80 if org == "TRUST_A" else 0.76):
                los = max(0.5, float(rng.normal(mean_los, 2.1)))
                discharge = min(admit + timedelta(days=los), now)
            else:
                discharge = None
            ip.append({"org_id": org, "admit": admit, "discharge": discharge})
    ip = pd.DataFrame(ip)

    # Theatres cancellation rate (28d)
    th = []
    for org in ["TRUST_A", "TRUST_B"]:
        cancel_rate = 0.09 if org == "TRUST_A" else 0.06
        for _ in range(550):
            start = now - timedelta(days=int(rng.integers(0, 40)), hours=int(rng.integers(0, 11)))
            status = "CANCELLED" if rng.random() < cancel_rate else "COMPLETED"
            th.append({"org_id": org, "scheduled_start": start, "status": status})
    th = pd.DataFrame(th)

    # Imaging TAT hours (7d)
    img = []
    for org in ["TRUST_A", "TRUST_B"]:
        mean_tat = 22 if org == "TRUST_A" else 16
        for _ in range(750):
            ordered = now - timedelta(days=int(rng.integers(0, 10)), hours=int(rng.integers(0, 24)))
            if rng.random() < 0.86:
                tat_h = max(0.5, float(rng.normal(mean_tat, 8)))
                reported = min(ordered + timedelta(hours=tat_h), now)
            else:
                reported = None
            img.append({"org_id": org, "ordered": ordered, "reported": reported})
    img = pd.DataFrame(img)

    # Workforce fill rate (7d)
    wf = []
    for org in ["TRUST_A", "TRUST_B"]:
        fill = 0.90 if org == "TRUST_A" else 0.94
        for i in range(14 * 3 * 4):
            start = now - timedelta(hours=8 * i)
            wf.append({"org_id": org, "ts": start, "is_filled": bool(rng.random() < fill)})
    wf = pd.DataFrame(wf)

    return beds, ed, ip, th, img, wf

def init_state():
    if "data" not in st.session_state:
        st.session_state["data"] = _seed_data()
    if "tasks" not in st.session_state:
        st.session_state["tasks"] = []
    if "approvals" not in st.session_state:
        st.session_state["approvals"] = []
    if "executions" not in st.session_state:
        st.session_state["executions"] = []

init_state()
beds_df, ed_df, ip_df, th_df, img_df, wf_df = st.session_state["data"]

# ============================
# KPI CALCS
# ============================
def kpi_bed_occupancy(org):
    df = beds_df[beds_df.org_id == org]
    return None if df.empty else float(df.is_occupied.mean())

def kpi_avg_los_28d(org):
    df = ip_df[(ip_df.org_id == org) & (ip_df.discharge.notna())]
    if df.empty:
        return None
    recent = df[df.discharge >= (datetime.utcnow() - timedelta(days=28))]
    use = recent if not recent.empty else df
    los = (use.discharge - use.admit).dt.total_seconds() / 86400.0
    return float(los.mean())

def kpi_theatre_cancel_rate_28d(org):
    df = th_df[(th_df.org_id == org) & (th_df.scheduled_start >= (datetime.utcnow() - timedelta(days=28)))]
    return None if df.empty else float((df.status == "CANCELLED").mean())

def kpi_imaging_tat_hours_7d(org):
    df = img_df[
        (img_df.org_id == org)
        & (img_df.ordered >= (datetime.utcnow() - timedelta(days=7)))
        & (img_df.reported.notna())
    ]
    if df.empty:
        return None
    tat = (df.reported - df.ordered).dt.total_seconds() / 3600.0
    return float(tat.mean())

def kpi_shift_fill_rate_7d(org):
    df = wf_df[(wf_df.org_id == org) & (wf_df.ts >= (datetime.utcnow() - timedelta(days=7)))]
    return None if df.empty else float(df.is_filled.mean())

def ed_series(org):
    return ed_df[ed_df.org_id == org].copy().sort_values("ts")

def forecast_next_24h(org):
    s = ed_series(org)["value"].to_numpy()
    if len(s) < 8:
        return [float(s[-1])] * 24 if len(s) else []
    y = s[-72:].astype(float)
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    last = x[-1]
    return [float(m * (last + h) + c) for h in range(1, 25)]

def avg_safe(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else float(np.mean(vals))

# ============================
# ALERTS / TASKS
# ============================
ALERTS = [
    {
        "key": "bed_occupancy_critical",
        "severity": "CRITICAL",
        "check": lambda org: (kpi_bed_occupancy(org) is not None and kpi_bed_occupancy(org) > 0.92),
        "title": "Escalation capacity decision",
        "guidance": "Review escalation beds 12–36h; confirm staffing; prioritise discharge push.",
        "recommend": lambda org: [
            {
                "action_key": "open_escalation_beds",
                "parameters": {"ward": "AMU", "beds": 8, "hours": 24},
                "expected_impact": "Reduce boarding pressure; protect ED performance.",
                "risk": "Requires staffing confirmation and site safety checks.",
            }
        ],
    },
    {
        "key": "imaging_tat_high",
        "severity": "HIGH",
        "check": lambda org: (kpi_imaging_tat_hours_7d(org) is not None and kpi_imaging_tat_hours_7d(org) > 24),
        "title": "Diagnostics backlog response",
        "guidance": "Review modality bottleneck; reporting capacity; prioritisation rules.",
        "recommend": lambda org: [],
    },
    {
        "key": "shift_fill_rate_low",
        "severity": "HIGH",
        "check": lambda org: (kpi_shift_fill_rate_7d(org) is not None and kpi_shift_fill_rate_7d(org) < 0.92),
        "title": "Staffing mitigation required",
        "guidance": "Review shortfall hotspots; release bank requests; consider redeployment.",
        "recommend": lambda org: [
            {
                "action_key": "release_bank_requests",
                "parameters": {"unit": "ED", "shifts": 3},
                "expected_impact": "Mitigate shortfalls in high acuity area.",
                "risk": "May increase temporary staffing cost.",
            }
        ],
    },
]

def run_alerts(org):
    now = datetime.utcnow()
    created = 0
    for a in ALERTS:
        if a["check"](org):
            recent = [
                t for t in st.session_state["tasks"]
                if t["org_id"] == org and t["alert_key"] == a["key"] and t["status"] == "OPEN"
                and (now - t["created_ts"]).total_seconds() < 6 * 3600
            ]
            if recent:
                continue
            st.session_state["tasks"].insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "org_id": org,
                    "created_ts": now,
                    "alert_key": a["key"],
                    "title": a["title"],
                    "severity": a["severity"],
                    "status": "OPEN",
                    "guidance": a["guidance"],
                    "recommendation": {"actions": a["recommend"](org)},
                },
            )
            created += 1
    return created

# ============================
# BOARD PACK PDF
# ============================
def board_pack_pdf(org):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"NHS Atlas Board Pack (Cloud Demo) — {org}")
    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 22

    kpis = {
        "Bed occupancy": kpi_bed_occupancy(org),
        "Avg LOS (28d) days": kpi_avg_los_28d(org),
        "Theatre cancel rate (28d)": kpi_theatre_cancel_rate_28d(org),
        "Imaging TAT (7d) hours": kpi_imaging_tat_hours_7d(org),
        "Shift fill rate (7d)": kpi_shift_fill_rate_7d(org),
    }

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Key KPIs")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in kpis.items():
        if v is None:
            vv = "n/a"
        elif "rate" in k.lower() or "occupancy" in k.lower() or "fill" in k.lower():
            vv = f"{v*100:.1f}%"
        else:
            vv = f"{v:.2f}"
        c.drawString(60, y, f"- {k}: {vv}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Open Tasks")
    y -= 18
    c.setFont("Helvetica", 10)
    tasks = [t for t in st.session_state["tasks"] if t["org_id"] == org and t["status"] == "OPEN"][:15]
    if not tasks:
        c.drawString(60, y, "No open tasks.")
    else:
        for t in tasks:
            c.drawString(60, y, f"- [{t['severity']}] {t['title']} ({t['alert_key']})")
            y -= 14
            if y < 90:
                c.showPage()
                y = h - 60
                c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return buf.getvalue()

# ============================
# UI
# ============================
st.title("NHS Atlas ULTRA — Cloud (Password Protected)")

with st.sidebar:
    st.header("Organisation")
    org = st.selectbox("Organisation", ORG_LIST, index=0)

    if st.button("Refresh (run alerts)"):
        if org in ["TRUST_A", "TRUST_B"]:
            created = run_alerts(org)
            st.success(f"Refreshed. New tasks: {created}")
        else:
            st.info("Select a Trust to run alerts in this demo.")

    pdf = board_pack_pdf(org if org != "ICB_DEMO" else "TRUST_A")
    st.download_button("Download Board Pack (PDF)", data=pdf, file_name=f"board_pack_{org}.pdf", mime="application/pdf")

orgs = ["TRUST_A", "TRUST_B"] if org == "ICB_DEMO" else [org]

occ = avg_safe([kpi_bed_occupancy(o) for o in orgs])
los = avg_safe([kpi_avg_los_28d(o) for o in orgs])
can = avg_safe([kpi_theatre_cancel_rate_28d(o) for o in orgs])
imt = avg_safe([kpi_imaging_tat_hours_7d(o) for o in orgs])
fil = avg_safe([kpi_shift_fill_rate_7d(o) for o in orgs])

cols = st.columns(5)
cols[0].metric("Bed occupancy", "n/a" if occ is None else f"{occ*100:.1f}%")
cols[1].metric("Avg LOS (28d)", "n/a" if los is None else f"{los:.2f} d")
cols[2].metric("Theatre cancels (28d)", "n/a" if can is None else f"{can*100:.1f}%")
cols[3].metric("Imaging TAT (7d)", "n/a" if imt is None else f"{imt:.1f} h")
cols[4].metric("Shift fill (7d)", "n/a" if fil is None else f"{fil*100:.1f}%")

st.subheader("ED Arrivals (Hourly) + Forecast")
if org == "ICB_DEMO":
    sA = ed_series("TRUST_A").set_index("ts")["value"]
    sB = ed_series("TRUST_B").set_index("ts")["value"]
    df_chart = pd.DataFrame({"TRUST_A": sA, "TRUST_B": sB}).fillna(0)
else:
    df_chart = ed_series(org).set_index("ts")[["value"]].rename(columns={"value": org})
st.line_chart(df_chart)

if org in ["TRUST_A", "TRUST_B"]:
    st.caption("Forecast (demo): simple linear trend next 24h.")
    st.write(forecast_next_24h(org))

st.subheader("Tasks / Approvals / Executions")
left, right = st.columns([1.3, 1])

with left:
    tasks = [t for t in st.session_state["tasks"] if t["org_id"] in orgs]
    if tasks:
        tdf = pd.DataFrame(tasks)
        tdf["created_ts"] = tdf["created_ts"].astype(str)
        st.dataframe(tdf, use_container_width=True)
    else:
        st.write("No tasks yet. Click **Refresh (run alerts)** in the sidebar.")

with right:
    st.markdown("### Create approval")
    open_tasks = [t for t in st.session_state["tasks"] if t["org_id"] in orgs and t["status"] == "OPEN"]
    pick = st.selectbox(
        "Pick task",
        options=["(none)"] + [f"{t['id']} — {t['title']} [{t['severity']}]" for t in open_tasks],
    )
    action = st.selectbox("Action", ["open_escalation_beds", "release_bank_requests", "notify_teams"])
    params_text = st.text_area("Parameters (JSON)", value='{"ward":"AMU","beds":8,"hours":24}')

    if st.button("Create approval request"):
        if pick == "(none)":
            st.warning("Select a task first.")
        else:
            try:
                _ = json.loads(params_text)
            except Exception:
                st.error("Parameters must be valid JSON.")
            else:
                tid = pick.split(" — ")[0]
                st.session_state["approvals"].insert(
                    0,
                    {
                        "id": str(uuid.uuid4())[:8],
                        "org_id": orgs[0],
                        "task_id": tid,
                        "action_key": action,
                        "parameters": params_text,
                        "status": "PENDING",
                        "created_ts": datetime.utcnow(),
                    },
                )
                st.success("Approval created (pending).")

    st.markdown("### Approvals")
    appr = [a for a in st.session_state["approvals"] if a["org_id"] in orgs]
    if appr:
        adf = pd.DataFrame(appr)
        adf["created_ts"] = adf["created_ts"].astype(str)
        st.dataframe(adf, use_container_width=True)
    else:
        st.write("No approvals yet.")

    st.markdown("### Execute (demo)")
    appr_ids = [a["id"] for a in appr] if appr else []
    aid = st.selectbox("Approval ID", options=["(none)"] + appr_ids)

    if st.button("Approve + Execute"):
        if aid == "(none)":
            st.warning("Choose an approval id.")
        else:
            for a in st.session_state["approvals"]:
                if a["id"] == aid:
                    a["status"] = "EXECUTED"
            st.session_state["executions"].insert(
                0,
                {
                    "id": str(uuid.uuid4())[:8],
                    "approval_id": aid,
                    "status": "SENT",
                    "created_ts": datetime.utcnow(),
                    "result": "Recorded (demo). Integrate real systems to enact changes.",
                },
            )
            st.success("Executed (recorded).")

    st.markdown("### Executions")
    ex = st.session_state["executions"]
    if ex:
        edf = pd.DataFrame(ex)
        edf["created_ts"] = edf["created_ts"].astype(str)
        st.dataframe(edf, use_container_width=True)
    else:
        st.write("No executions yet.")

st.subheader("Ask Atlas (Copilot)")
q = st.text_input("Question (no patient identifiers):", value="What are today’s biggest risks and what should we do first?")
if st.button("Ask Copilot"):
    open_tasks = [t for t in st.session_state["tasks"] if t["org_id"] in orgs and t["status"] == "OPEN"][:5]
    ans = []
    ans.append("Atlas Copilot (safe mode)")
    ans.append(f"Question: {q}")
    ans.append("Top open tasks:")
    if open_tasks:
        for t in open_tasks:
            ans.append(f"- [{t['severity']}] {t['title']} ({t['alert_key']})")
    else:
        ans.append("- None yet (click Refresh to generate alerts).")
    ans.append("Suggested order:")
    ans.append("1) Address CRITICAL bed occupancy / staffing shortfalls first.")
    ans.append("2) Stabilise theatres and diagnostics turnaround.")
    ans.append("3) Confirm data quality if anomaly alerts are sustained.")
    st.text("\n".join(ans))

st.caption("Cloud demo: no local installs. For production, connect Trust/ICB data feeds and add SSO + IG controls.")
