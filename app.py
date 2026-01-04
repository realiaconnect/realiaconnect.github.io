import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import uuid
import json
import re

# =========================================================
# PASSWORD (simple demo gate)
# =========================================================
PASSWORD_HASH = hashlib.sha256("enflowers".encode()).hexdigest()

def check_password():
    def password_entered():
        ok = hashlib.sha256(st.session_state["password"].encode()).hexdigest() == PASSWORD_HASH
        st.session_state["password_correct"] = ok
        del st.session_state["password"]

    if "password_correct" not in st.session_state:
        st.text_input("🔐 Enter password", type="password", on_change=password_entered, key="password")
        return False
    if not st.session_state["password_correct"]:
        st.text_input("🔐 Enter password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False
    return True

st.set_page_config(page_title="NHS ULTRA OS — Hospital (Synthetic)", layout="wide")
if not check_password():
    st.stop()

# =========================================================
# SPECIALTY FACTORY (200+)
# =========================================================
BASE_SPECIALTIES = [
    "Emergency Medicine","Acute Medicine","General Medicine","Respiratory","Gastroenterology","Hepatology","Renal",
    "Endocrinology","Diabetes","Cardiology","Heart Failure","Interventional Cardiology","Electrophysiology",
    "Haematology","Oncology","Radiotherapy","Infectious Diseases","Microbiology","Rheumatology","Immunology",
    "Dermatology","Geriatrics","Stroke","Neurology","Epilepsy","Neurosurgery",
    "General Surgery","Trauma & Orthopaedics","Spinal Surgery","Vascular Surgery","Urology","ENT","Ophthalmology",
    "Plastics","Maxillofacial","Transplant","Bariatric Surgery",
    "Obstetrics","Maternity","Gynaecology","Neonatology","NICU","Paediatrics","PICU","Paediatric Surgery",
    "ICU","HDU","Anaesthetics","Pain Service",
    "Psychiatry","Mental Health Liaison","Substance Misuse",
    "Radiology","Interventional Radiology","Pathology","Cardiac Physiology","Endoscopy",
    "Physiotherapy","Occupational Therapy","Speech & Language","Dietetics","Pharmacy"
]
SUBSPECIALTY_BUCKETS = {
    "Cardiology": ["ACS", "Chest Pain", "Arrhythmia", "Heart Failure", "Structural", "PCI Pathway"],
    "Neurology": ["Stroke", "Seizure", "Headache", "Neuroinflammation", "Neuro ICU Liaison"],
    "Respiratory": ["COPD", "Asthma", "Pneumonia", "Pleural", "Sleep", "Resp Failure"],
    "Oncology": ["2WW", "Staging", "Chemo Day Unit", "RT Pathway", "MDT"],
    "Surgery": ["Elective", "Emergency", "Trauma", "Day Case", "Pre-op Optimisation"],
    "Paediatrics": ["Peds ED", "RTI Surge", "Safeguarding", "NICU Liaison", "PICU Liaison"],
    "Maternity": ["Triage", "Labour Ward", "Postnatal", "High Risk", "Induction"],
    "ICU": ["Ventilation", "Step-down", "Sepsis Support", "Outreach", "Deterioration"],
}
def build_specialties():
    out = []
    out.extend(BASE_SPECIALTIES)
    for k, subs in SUBSPECIALTY_BUCKETS.items():
        for s in subs:
            out.append(f"{k} — {s}")
    for s in ["Diagnostics","Operations","Flow Hub","Discharge Team","Site Management","Bed Management","Theatres Management"]:
        out.append(s)
    variants = ["Pathway","Clinic","MDT","Ward Round","Hotline","Rapid Review","Virtual Ward"]
    for base in BASE_SPECIALTIES[:45]:
        for v in variants:
            out.append(f"{base} — {v}")
    out = list(dict.fromkeys(out))
    return out

SPECIALTIES = build_specialties()

# =========================================================
# PATHWAY ENGINE (hospital-only demo)
# =========================================================
PATHWAYS = {
    "Sepsis": {
        "signals": ["Lactate Critical", "NEWS High", "BP Low", "WBC High", "CRP High"],
        "actions": ["Sepsis 6 bundle prompt", "Senior review", "Blood cultures", "ABx within 1 hour", "Fluids + reassess"],
        "owners": ["ED", "AMU", "ICU Outreach"]
    },
    "Stroke": {
        "signals": ["CT Pending", "FAST Positive", "Door-to-needle Delayed"],
        "actions": ["Stroke team activation", "CT fast-track", "Thrombolysis checklist", "Transfer to stroke bed"],
        "owners": ["ED", "Stroke", "Radiology"]
    },
    "ACS": {
        "signals": ["Troponin Critical", "Chest pain queue high"],
        "actions": ["ECG within 10 min", "Cardio review", "Cath lab prioritisation", "Critical result acknowledged"],
        "owners": ["ED", "Cardiology"]
    },
    "AKI": {
        "signals": ["Creatinine High", "K High"],
        "actions": ["AKI bundle", "Fluid status review", "Renal referral rules", "Medication hold suggestions"],
        "owners": ["AMU", "Renal", "Pharmacy"]
    },
    "COPD": {
        "signals": ["Resp failure risk", "Neb backlog"],
        "actions": ["Resp pathway", "ABG if indicated", "Early supported discharge check"],
        "owners": ["ED", "Respiratory"]
    },
    "DKA": {
        "signals": ["Glucose High", "Ketones High"],
        "actions": ["DKA insulin protocol prompt", "Hourly monitoring", "Endocrine review"],
        "owners": ["ED", "AMU", "Endocrinology"]
    },
    "Maternity triage": {
        "signals": ["Triage wait high", "Delivery suite occupancy high"],
        "actions": ["Triage surge staffing", "Fast-track protocol", "Escalation to coordinator"],
        "owners": ["Maternity", "Anaesthetics"]
    }
}

# =========================================================
# HOSPITAL SYNTHETIC MODEL
# =========================================================
TRUSTS = ["TRUST_A", "TRUST_B"]
SITES = ["City Hospital", "General Hospital"]
WARDS = ["ED", "AMU", "SAU", "Respiratory", "Cardiology", "Neuro", "Stroke", "Surgery", "Oncology", "Maternity", "ICU", "Paediatrics"]
BAYS = ["A","B","C","D"]
BED_STATES = ["Occupied","Vacant","Cleaning","Blocked"]

ROLES = ["Clinician","Ops","Pharmacy","Manager"]

def redact_text(s: str) -> str:
    s = re.sub(r"\bP\d{5}\b", "[PATIENT_ID]", s)
    s = re.sub(r"\b\d{3}\s\d{3}\s\d{4}\b", "[NHS_ID]", s)
    return s

def init_state():
    if "demo" not in st.session_state:
        st.session_state["demo"] = seed_demo()
    if "tasks" not in st.session_state:
        st.session_state["tasks"] = []
    if "approvals" not in st.session_state:
        st.session_state["approvals"] = []
    if "executions" not in st.session_state:
        st.session_state["executions"] = []
    if "audit" not in st.session_state:
        st.session_state["audit"] = []
    if "selected_patient_id" not in st.session_state:
        st.session_state["selected_patient_id"] = None

def audit(event, details):
    st.session_state["audit"].insert(0, {
        "ts": datetime.utcnow().isoformat(timespec="minutes"),
        "event": event,
        "details": details[:4000]
    })

def seed_demo(n_patients=140):
    rng = np.random.default_rng(2026)
    now = datetime.utcnow()

    beds = []
    for trust in TRUSTS:
        for site in SITES:
            for ward in WARDS:
                for bay in BAYS:
                    for bed_num in range(1, 13):
                        beds.append({
                            "trust": trust, "site": site, "ward": ward, "bay": bay, "bed": f"{bay}{bed_num}",
                            "state": rng.choice(BED_STATES, p=[0.72,0.18,0.07,0.03]),
                            "patient_id": None,
                            "edd": None,
                            "updated_ts": now.isoformat(timespec="minutes")
                        })
    bedboard = pd.DataFrame(beds)

    first_names = ["Alex","Sam","Jordan","Taylor","Casey","Jamie","Morgan","Riley","Avery","Robin","Cameron","Drew","Amina","Noah","Luca","Maya","Sofia","Mateo"]
    last_names = ["Smith","Jones","Brown","Taylor","Wilson","Johnson","Davies","Evans","Thomas","Roberts","Walker","Hall","Khan","Patel","Lewis","Clark","Lopez","Garcia"]

    occupied_idx = bedboard.index[bedboard["state"]=="Occupied"].to_list()
    rng.shuffle(occupied_idx)
    occupied_idx = occupied_idx[:n_patients]

    pts = []
    for i in range(n_patients):
        pid = f"P{10000+i}"
        row_idx = occupied_idx[i]
        b = bedboard.loc[row_idx].to_dict()

        if b["ward"] in ["Stroke","Neuro"]:
            spec = rng.choice([s for s in SPECIALTIES if ("Neuro" in s or "Stroke" in s or "Neurology" in s or "Epilepsy" in s)] + ["Stroke","Neurology","Neurosurgery"])
        elif b["ward"] == "Cardiology":
            spec = rng.choice([s for s in SPECIALTIES if "Cardio" in s] + ["Cardiology","Heart Failure"])
        elif b["ward"] == "Paediatrics":
            spec = rng.choice([s for s in SPECIALTIES if ("Paed" in s or "Peds" in s or "NICU" in s or "PICU" in s)] + ["Paediatrics","PICU"])
        elif b["ward"] == "ICU":
            spec = rng.choice([s for s in SPECIALTIES if "ICU" in s] + ["ICU","Anaesthetics"])
        else:
            spec = rng.choice(SPECIALTIES)

        dob_year = int(rng.integers(1935, 2024))
        dob = datetime(dob_year, int(rng.integers(1,13)), int(rng.integers(1,29)))
        admit = now - timedelta(days=int(rng.integers(0,21)), hours=int(rng.integers(0,24)))
        edd = (admit + timedelta(days=float(np.clip(rng.normal(6, 3), 1, 18)))).date().isoformat()

        risk = rng.choice(["None","High","Critical"], p=[0.70,0.23,0.07])
        barrier = rng.choice(
            ["None","Awaiting imaging","Awaiting PT/OT","Awaiting meds","Awaiting social care","Awaiting consultant review"],
            p=[0.28,0.18,0.14,0.12,0.16,0.12]
        )

        pts.append({
            "patient_id": pid,
            "name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
            "dob": dob.date().isoformat(),
            "nhs_like_id": f"{int(rng.integers(100,999))} {int(rng.integers(100,999))} {int(rng.integers(1000,9999))}",
            "trust": b["trust"], "site": b["site"], "ward": b["ward"], "bay": b["bay"], "bed": b["bed"],
            "specialty": spec,
            "status": rng.choice(["Inpatient","ED","Outpatient"], p=[0.85,0.10,0.05]),
            "risk_flag": risk,
            "admit_ts": admit.isoformat(timespec="minutes"),
            "edd": edd,
            "discharge_barriers": barrier,
        })
        bedboard.loc[row_idx, "patient_id"] = pid
        bedboard.loc[row_idx, "edd"] = edd

    patients = pd.DataFrame(pts)

    meds_catalog = [
        ("Paracetamol","1g PO QDS"), ("Ibuprofen","400mg PO TDS"),
        ("Amoxicillin","500mg PO TDS"), ("Metformin","500mg PO BD"),
        ("Atorvastatin","20mg PO nocte"), ("Amlodipine","5mg PO OD"),
        ("Salbutamol","2 puffs PRN"), ("Insulin","as per sliding scale"),
        ("Heparin","prophylactic dose"), ("Omeprazole","20mg PO OD"),
        ("Morphine","as prescribed"), ("Furosemide","40mg PO OD"),
        ("Apixaban","5mg PO BD"), ("Prednisolone","as prescribed"),
    ]
    meds = []
    for pid in patients["patient_id"]:
        for _ in range(int(rng.integers(2,7))):
            drug, sig = meds_catalog[int(rng.integers(0, len(meds_catalog)))]
            meds.append({
                "patient_id": pid, "drug": drug, "dose": sig,
                "start_date": (now - timedelta(days=int(rng.integers(0,25)))).date().isoformat(),
                "status": rng.choice(["Active","Held","Stopped"], p=[0.78,0.12,0.10])
            })
    meds = pd.DataFrame(meds)

    labs = ["Hb","WBC","CRP","Na","K","Creatinine","Glucose","Troponin","Lactate","Ketones"]
    results = []
    for pid in patients["patient_id"]:
        for _ in range(int(rng.integers(6,20))):
            lab = rng.choice(labs)
            ts = (now - timedelta(hours=int(rng.integers(2,240)))).replace(minute=0, second=0, microsecond=0)
            value = float(rng.normal(10, 3))
            flag = rng.choice(["Normal","High","Low","Critical"], p=[0.70,0.16,0.12,0.02])
            if lab in ["Troponin","Lactate","K","Glucose","Ketones"] and rng.random() < 0.05:
                flag = "Critical"
            results.append({"patient_id": pid, "ts": ts.isoformat(), "test": lab, "value": value, "flag": flag})
    results = pd.DataFrame(results).sort_values(["patient_id","ts"], ascending=[True, False])

    modalities = ["XR","CT","MRI","US"]
    imaging = []
    for pid in patients["patient_id"]:
        for _ in range(int(rng.integers(0,4))):
            modality = rng.choice(modalities)
            d = (now - timedelta(days=int(rng.integers(0,30)))).date().isoformat()
            imaging.append({
                "patient_id": pid, "date": d, "modality": modality,
                "study": f"{modality} {rng.choice(['Head','Chest','Abdo','Spine','Pelvis','Knee'])}",
                "report_status": rng.choice(["Reported","Pending","In progress"], p=[0.62,0.25,0.13]),
                "report_excerpt": rng.choice([
                    "No acute abnormality identified.",
                    "Findings consistent with infection/inflammation.",
                    "Further correlation recommended.",
                    "Significant finding; clinician informed."
                ])
            })
    imaging = pd.DataFrame(imaging)

    appts = []
    theatres = []
    for pid in patients["patient_id"]:
        for _ in range(int(rng.integers(0,3))):
            dt = now + timedelta(days=int(rng.integers(1,60)))
            appts.append({
                "patient_id": pid,
                "date": dt.date().isoformat(),
                "time": f"{int(rng.integers(8,17)):02d}:{rng.choice([0,15,30,45]):02d}",
                "clinic": rng.choice(["Cardiology OPD","Neuro OPD","Respiratory OPD","Surgical FU","Diabetes Clinic","Oncology Review"]),
                "status": rng.choice(["Booked","Cancelled","DNA risk"], p=[0.80,0.08,0.12])
            })
        if rng.random() < 0.10:
            dt = now + timedelta(days=int(rng.integers(1,20)))
            theatres.append({
                "patient_id": pid,
                "date": dt.date().isoformat(),
                "list": f"Theatre {int(rng.integers(1,6))}",
                "procedure": rng.choice(["Lap chole","ORIF","C-section","Appendectomy","Hernia repair","Angio"]),
                "status": rng.choice(["Scheduled","Cancelled risk","Completed"], p=[0.70,0.18,0.12])
            })

    appts = pd.DataFrame(appts)
    theatres = pd.DataFrame(theatres)

    ed = []
    hours = 14*24
    for trust in TRUSTS:
        base = 12 if trust=="TRUST_A" else 10
        for h in range(hours):
            t = now - timedelta(hours=(hours-1-h))
            baseline = base + 3*np.sin((h/24.0)*2*np.pi)
            arrivals = max(0, int(rng.normal(baseline, 3)))
            ed.append({"trust": trust, "ts": t.replace(minute=0, second=0, microsecond=0), "arrivals": arrivals})
    ed = pd.DataFrame(ed)

    return {"patients": patients, "bedboard": bedboard, "meds": meds, "results": results,
            "imaging": imaging, "appts": appts, "theatres": theatres, "ed": ed}

init_state()
DEMO = st.session_state["demo"]

# =========================================================
# AUTOMATION (safe demo "AI")
# =========================================================
def make_task(trust, title, severity, alert_key, guidance, owner_role="Ops", payload=None):
    t = {
        "id": str(uuid.uuid4())[:8],
        "trust": trust,
        "created_ts": datetime.utcnow().isoformat(timespec="minutes"),
        "title": title,
        "severity": severity,
        "alert_key": alert_key,
        "status": "OPEN",
        "owner_role": owner_role,
        "guidance": guidance,
        "payload": payload or {}
    }
    recent = [x for x in st.session_state["tasks"] if x["trust"]==trust and x["alert_key"]==alert_key and x["status"]=="OPEN"]
    if not recent:
        st.session_state["tasks"].insert(0, t)
        audit("task_created", f"{title} | {guidance}")
        return True
    return False

def run_autopilot(trust, site):
    created = 0
    pts = DEMO["patients"]
    bb = DEMO["bedboard"]

    bb_f = bb[(bb["trust"]==trust) & (bb["site"]==site)]
    occ = (bb_f["state"]=="Occupied").mean()
    if occ > 0.90:
        if make_task(trust, "Bed occupancy critical", "CRITICAL", f"bed_pressure::{site}",
                     f"Occupancy {occ*100:.1f}% at {site}. Trigger discharge push + escalation beds.", "Ops",
                     {"occ": occ, "site": site, "playbook": ["Discharge-by-noon push", "Open escalation beds", "Daily safety huddle"]}):
            created += 1

    icu_count = pts[(pts["trust"]==trust) & (pts["site"]==site) & (pts["ward"]=="ICU")].shape[0]
    rng = np.random.default_rng(int(datetime.utcnow().strftime("%H"))+7)
    staffing_fill = float(np.clip(rng.normal(0.93 - (icu_count/250), 0.03), 0.70, 0.99))
    if staffing_fill < 0.92:
        if make_task(trust, "ICU staffing mitigation required", "CRITICAL", f"icu_staff::{site}",
                     f"ICU staffing fill {staffing_fill*100:.1f}% with {icu_count} ICU patients. Release bank/agency + redeploy plan.",
                     "Manager", {"fill": staffing_fill, "icu_patients": icu_count, "site": site}):
            created += 1

    r = DEMO["results"]
    crit = r.merge(pts[["patient_id","trust","site"]], on="patient_id", how="left")
    crit = crit[(crit["trust"]==trust)&(crit["site"]==site)&(crit["flag"]=="Critical")]
    crit_count = crit.shape[0]
    if crit_count > 25:
        if make_task(trust, "Critical results backlog", "HIGH", f"crit_results::{site}",
                     f"Critical lab flags count={crit_count} (demo). Trigger acknowledgement sweep + escalation.",
                     "Clinician", {"count": crit_count, "site": site}):
            created += 1

    im = DEMO["imaging"].merge(pts[["patient_id","trust","site"]], on="patient_id", how="left")
    pend = im[(im["trust"]==trust)&(im["site"]==site)&(im["report_status"]!="Reported")].shape[0]
    if pend > 30:
        if make_task(trust, "Imaging reporting backlog", "HIGH", f"img_backlog::{site}",
                     f"Pending/in-progress imaging count={pend}. Consider reporting sprint + prioritisation rules.", "Ops",
                     {"pending": pend, "site": site}):
            created += 1

    barriers = pts[(pts["trust"]==trust)&(pts["site"]==site) & (pts["discharge_barriers"]!="None")].shape[0]
    if barriers > 55:
        if make_task(trust, "Discharge barriers high", "HIGH", f"discharge_barriers::{site}",
                     f"Patients with barriers={barriers}. Run MDT discharge huddle + targeted actions.", "Ops",
                     {"barriers": barriers, "site": site}):
            created += 1

    return created, {"occ": float(occ), "icu_fill": float(staffing_fill), "crit": int(crit_count), "img_pending": int(pend), "barriers": int(barriers)}

# =========================================================
# HELPERS
# =========================================================
def ed_series(trust):
    d = DEMO["ed"][DEMO["ed"]["trust"]==trust].copy().sort_values("ts")
    return d.set_index("ts")["arrivals"]

def patient_df_filtered(trust, site, ward="All", specialty="All", query=""):
    df = DEMO["patients"].copy()
    df = df[(df["trust"]==trust) & (df["site"]==site)]
    if ward != "All":
        df = df[df["ward"]==ward]
    if specialty != "All":
        df = df[df["specialty"]==specialty]
    if query.strip():
        q = query.strip().lower()
        df = df[df["name"].str.lower().str.contains(q) | df["patient_id"].str.lower().str.contains(q)]
    return df

def board_pack_text(trust, site, role):
    bb = DEMO["bedboard"]
    pts = DEMO["patients"]
    bb_f = bb[(bb["trust"]==trust)&(bb["site"]==site)]
    occ = (bb_f["state"]=="Occupied").mean()
    vac = (bb_f["state"]=="Vacant").sum()
    crit = DEMO["results"].merge(pts[["patient_id","trust","site"]], on="patient_id", how="left")
    crit = crit[(crit["trust"]==trust)&(crit["site"]==site)&(crit["flag"]=="Critical")].shape[0]

    lines = []
    lines.append(f"NHS ULTRA OS — Board Pack (Synthetic Demo) — {trust} / {site}")
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
    lines.append(f"Role view: {role}")
    lines.append("")
    lines.append("Capacity")
    lines.append(f"- Bed occupancy: {occ*100:.1f}%")
    lines.append(f"- Vacant beds: {vac}")
    lines.append("")
    lines.append("Clinical safety")
    lines.append(f"- Critical lab flags (count): {crit}")
    lines.append("")
    lines.append("Open tasks (top 15)")
    tasks = [t for t in st.session_state["tasks"] if t["trust"]==trust and t["status"]=="OPEN"][:15]
    if not tasks:
        lines.append("- None")
    else:
        for t in tasks:
            lines.append(f"- [{t['severity']}] {t['title']} ({t['alert_key']}) owner={t['owner_role']}")
    lines.append("")
    lines.append("Note: Demo uses synthetic patients. Production: integrate EPR/FHIR + RBAC + audit + IG controls.")
    return "\n".join(lines)

# =========================================================
# UI
# =========================================================
st.title("NHS ULTRA OS — Hospital-only (Synthetic Demo)")

with st.sidebar:
    st.header("Session")
    role = st.selectbox("Role", ROLES, index=0)
    trust = st.selectbox("Trust", TRUSTS, index=0)
    site = st.selectbox("Site", SITES, index=0)

    st.divider()
    if st.button("⚡ Run Autopilot (AI demo)"):
        created, snapshot = run_autopilot(trust, site)
        st.success(f"Autopilot ran. New tasks: {created}")
        audit("autopilot_run", json.dumps(snapshot))

    bp = board_pack_text(trust, site, role)
    st.download_button("Download Board Pack (TXT)", data=bp.encode("utf-8"), file_name=f"board_pack_{trust}_{site}.txt", mime="text/plain")

    st.caption("Safety: Synthetic data only. No real patient data stored.")

tabs = st.tabs([
    "Command Centre", "Bedboard Live", "Patient Workspace", "Appointments & Theatres",
    "Results & Imaging", "Pathways", "Tasks & Approvals", "Audit"
])

# ---------------------------------------------------------
# Command Centre
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Command Centre")
    bb = DEMO["bedboard"]
    bb_f = bb[(bb["trust"]==trust)&(bb["site"]==site)]
    occ_rate = (bb_f["state"]=="Occupied").mean()
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Total beds", len(bb_f))
    col2.metric("Occupied", int((bb_f["state"]=="Occupied").sum()))
    col3.metric("Vacant", int((bb_f["state"]=="Vacant").sum()))
    col4.metric("Cleaning", int((bb_f["state"]=="Cleaning").sum()))
    col5.metric("Blocked", int((bb_f["state"]=="Blocked").sum()))
    st.caption(f"Occupancy: {occ_rate*100:.1f}%")

    st.markdown("### ED arrivals (14 days hourly)")
    st.line_chart(ed_series(trust).to_frame(name=f"{trust} arrivals"))

    st.markdown("### Discharge barriers snapshot")
    pts = DEMO["patients"]
    psite = pts[(pts["trust"]==trust)&(pts["site"]==site)]
    bar = psite["discharge_barriers"].value_counts().reset_index()
    bar.columns = ["barrier","count"]
    st.dataframe(bar, use_container_width=True, height=220)

# ---------------------------------------------------------
# Bedboard Live
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Bedboard Live (synthetic)")
    ward_pick = st.selectbox("Ward", ["All"] + WARDS, index=0, key="ward_pick_bb")
    bb = DEMO["bedboard"][(DEMO["bedboard"]["trust"]==trust)&(DEMO["bedboard"]["site"]==site)].copy()
    if ward_pick != "All":
        bb = bb[bb["ward"]==ward_pick]

    pts = DEMO["patients"][["patient_id","name","specialty","risk_flag","edd","discharge_barriers"]]
    bb = bb.merge(pts, on="patient_id", how="left")

    # ✅ FIX: only select columns that exist (prevents KeyError)
    cols = ["ward","bay","bed","state","patient_id","name","specialty","risk_flag","edd","discharge_barriers","updated_ts"]
    cols = [c for c in cols if c in bb.columns]
    st.dataframe(bb[cols], use_container_width=True, height=420)

    st.markdown("### Transfer patient (demo)")
    cA,cB,cC = st.columns(3)
    pid = cA.text_input("Patient ID", value="")
    to_ward = cB.selectbox("To ward", WARDS, index=0, key="to_ward")
    to_bed = cC.text_input("To bed (e.g., A1)", value="A1")

    if st.button("Move patient"):
        if not pid or pid not in DEMO["patients"]["patient_id"].values:
            st.error("Enter a valid patient ID from Patient Workspace.")
        else:
            cur = DEMO["bedboard"][(DEMO["bedboard"]["trust"]==trust)&(DEMO["bedboard"]["site"]==site)&(DEMO["bedboard"]["patient_id"]==pid)]
            tgt = DEMO["bedboard"][(DEMO["bedboard"]["trust"]==trust)&(DEMO["bedboard"]["site"]==site)&(DEMO["bedboard"]["ward"]==to_ward)&(DEMO["bedboard"]["bed"]==to_bed)]
            if cur.empty:
                st.error("Patient not currently assigned in this trust/site.")
            elif tgt.empty:
                st.error("Target bed not found.")
            elif tgt.iloc[0]["state"]!="Vacant":
                st.error("Target bed must be Vacant.")
            else:
                cur_idx = cur.index[0]
                tgt_idx = tgt.index[0]
                DEMO["bedboard"].loc[cur_idx, ["state","patient_id","edd","updated_ts"]] = ["Vacant", None, None, datetime.utcnow().isoformat(timespec="minutes")]
                DEMO["bedboard"].loc[tgt_idx, ["state","patient_id","updated_ts"]] = ["Occupied", pid, datetime.utcnow().isoformat(timespec="minutes")]
                DEMO["patients"].loc[DEMO["patients"]["patient_id"]==pid, ["ward","bed","bay","site"]] = [to_ward, to_bed, to_bed[:1], site]
                audit("transfer", f"{pid} moved to {to_ward} {to_bed} ({trust}/{site})")
                make_task(trust, f"Transfer executed (demo): {pid} -> {to_ward} {to_bed}", "INFO", f"transfer::{pid}", "Transfer recorded in synthetic bedboard.", owner_role="Ops")
                st.success("Transfer simulated.")

# ---------------------------------------------------------
# Patient Workspace
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Patient Workspace")
    ward = st.selectbox("Ward filter", ["All"]+WARDS, index=0, key="ward_filter_pts")
    spec = st.selectbox("Specialty filter", ["All"]+SPECIALTIES, index=0, key="spec_filter_pts")
    query = st.text_input("Search", value="", key="search_pts")

    df = patient_df_filtered(trust, site, ward, spec, query)
    st.dataframe(df[["patient_id","name","status","risk_flag","ward","bed","specialty","edd","discharge_barriers","admit_ts"]],
                 use_container_width=True, height=260)

    pid = st.text_input("Open patient by ID", value=st.session_state["selected_patient_id"] or "", key="open_pid")
    if pid and pid in DEMO["patients"]["patient_id"].values:
        st.session_state["selected_patient_id"] = pid
    pid = st.session_state["selected_patient_id"]

    if not pid:
        st.info("Open a patient to view meds/results/imaging.")
    else:
        p = DEMO["patients"][DEMO["patients"]["patient_id"]==pid].iloc[0].to_dict()
        a,b,c,d = st.columns(4)
        a.metric("Patient", pid)
        b.metric("Ward/Bed", f"{p['ward']} {p['bed']}")
        c.metric("EDD", p["edd"])
        d.metric("Barrier", p["discharge_barriers"])

        subtabs = st.tabs(["Meds","Results","Imaging","AI Copilot (safe demo)"])
        with subtabs[0]:
            if role not in ["Clinician","Pharmacy","Manager"]:
                st.warning("Role view: limited meds visibility.")
            m = DEMO["meds"][DEMO["meds"]["patient_id"]==pid]
            st.dataframe(m[["drug","dose","start_date","status"]], use_container_width=True)

        with subtabs[1]:
            r = DEMO["results"][DEMO["results"]["patient_id"]==pid]
            st.dataframe(r[["ts","test","value","flag"]], use_container_width=True, height=300)

        with subtabs[2]:
            im = DEMO["imaging"][DEMO["imaging"]["patient_id"]==pid]
            if im.empty:
                st.write("No imaging in demo.")
            else:
                st.dataframe(im[["date","modality","study","report_status","report_excerpt"]], use_container_width=True)
            st.info("Production: PACS/DICOM viewer integration + RBAC.")

        with subtabs[3]:
            st.write("This demo copilot is rule-based + redaction. Production AI requires IG controls and approval.")
            q = st.text_input("Ask", value="Summarise risks and next steps.", key="copilot_q")
            if st.button("Generate"):
                meds_active = DEMO["meds"][(DEMO["meds"]["patient_id"]==pid) & (DEMO["meds"]["status"]=="Active")].shape[0]
                crit = DEMO["results"][(DEMO["results"]["patient_id"]==pid) & (DEMO["results"]["flag"]=="Critical")].shape[0]
                pending_img = DEMO["imaging"][(DEMO["imaging"]["patient_id"]==pid) & (DEMO["imaging"]["report_status"]!="Reported")].shape[0]
                answer = f"""
AI Copilot (SAFE DEMO — no real AI calls)

Question: {q}

Snapshot:
- Location: {p['trust']} / {p['site']} / {p['ward']} bed {p['bed']}
- Specialty: {p['specialty']}
- Active meds: {meds_active}
- Critical lab flags: {crit}
- Imaging pending: {pending_img}
- Discharge barrier: {p['discharge_barriers']}

Suggested next steps:
- If critical results: acknowledge + escalate to senior review.
- If imaging pending: chase report; prioritise if deterioration.
- Address discharge barrier with targeted team action.
"""
                answer = redact_text(answer)
                st.text(answer)
                audit("copilot_used", answer)

# ---------------------------------------------------------
# Appointments & Theatres
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Appointments & Theatres")
    pid = st.session_state["selected_patient_id"]
    pts_site = DEMO["patients"][(DEMO["patients"]["trust"]==trust)&(DEMO["patients"]["site"]==site)][["patient_id"]]
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Appointments (synthetic)")
        ap = DEMO["appts"].merge(pts_site, on="patient_id", how="inner")
        st.dataframe(ap[["patient_id","date","time","clinic","status"]].sort_values(["date","time"]),
                     use_container_width=True, height=330)
    with c2:
        st.markdown("### Theatres (synthetic)")
        th = DEMO["theatres"]
        if th.empty:
            st.write("No theatre items in demo.")
        else:
            th = th.merge(pts_site, on="patient_id", how="inner")
            st.dataframe(th[["patient_id","date","list","procedure","status"]].sort_values(["date","list"]),
                         use_container_width=True, height=330)

# ---------------------------------------------------------
# Results & Imaging (site-wide)
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Results & Imaging (site-wide)")
    pts_site = DEMO["patients"][(DEMO["patients"]["trust"]==trust)&(DEMO["patients"]["site"]==site)][["patient_id","ward","specialty","risk_flag"]]

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Critical labs feed")
        r = DEMO["results"].merge(pts_site, on="patient_id", how="inner")
        r = r[r["flag"]=="Critical"].head(200)
        st.dataframe(r[["ts","patient_id","ward","test","value","flag","specialty","risk_flag"]], use_container_width=True, height=320)
    with c2:
        st.markdown("### Imaging pending feed")
        im = DEMO["imaging"].merge(pts_site, on="patient_id", how="inner")
        im = im[im["report_status"]!="Reported"].head(200)
        st.dataframe(im[["date","patient_id","ward","modality","study","report_status","specialty"]], use_container_width=True, height=320)

# ---------------------------------------------------------
# Pathways
# ---------------------------------------------------------
with tabs[5]:
    st.subheader("Pathways (hospital-only)")
    pathway = st.selectbox("Pathway", list(PATHWAYS.keys()), index=0)
    pw = PATHWAYS[pathway]

    st.markdown("### Signals")
    for s in pw["signals"]:
        st.write(f"• {s}")

    st.markdown("### Actions")
    for a in pw["actions"]:
        st.write(f"• {a}")

    st.markdown("### Owners")
    st.write(", ".join(pw["owners"]))

    st.markdown("### Generate pathway task (demo)")
    if st.button("Create pathway task"):
        created = make_task(
            trust,
            f"Pathway triggered: {pathway}",
            "HIGH",
            f"pathway::{pathway.lower()}::{site}",
            f"Trigger signals (demo): {', '.join(pw['signals'][:3])}…",
            owner_role="Clinician",
            payload={"pathway": pathway, "owners": pw["owners"], "actions": pw["actions"]}
        )
        if created:
            st.success("Task created.")
        else:
            st.info("Similar task already open (deduped).")

# ---------------------------------------------------------
# Tasks & Approvals
# ---------------------------------------------------------
with tabs[6]:
    st.subheader("Tasks & Approvals (demo)")
    tasks = [t for t in st.session_state["tasks"] if t["trust"]==trust]
    if tasks:
        st.dataframe(pd.DataFrame(tasks), use_container_width=True, height=260)
    else:
        st.info("No tasks yet. Run Autopilot or create a Pathway task.")

    st.markdown("### Create approval")
    open_tasks = [t for t in tasks if t["status"]=="OPEN"]
    pick = st.selectbox("Pick task", options=["(none)"] + [f"{t['id']} — {t['title']} [{t['severity']}]" for t in open_tasks])
    action = st.selectbox("Action", ["notify_teams","request_capacity","run_playbook","schedule_huddle","chase_results","expedite_imaging"])
    params_text = st.text_area("Parameters (JSON)", value=json.dumps({"site": site, "role": role, "priority": "HIGH"}))

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
                st.session_state["approvals"].insert(0, {
                    "id": str(uuid.uuid4())[:8],
                    "trust": trust,
                    "task_id": tid,
                    "action_key": action,
                    "parameters": params_text,
                    "status":"PENDING",
                    "created_ts": datetime.utcnow().isoformat(timespec="minutes")
                })
                audit("approval_created", f"task={tid} action={action} params={params_text}")
                st.success("Approval created (pending).")

    st.markdown("### Approvals")
    appr = [a for a in st.session_state["approvals"] if a["trust"]==trust]
    if appr:
        st.dataframe(pd.DataFrame(appr), use_container_width=True, height=200)
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
            st.session_state["executions"].insert(0, {
                "id": str(uuid.uuid4())[:8],
                "approval_id": aid,
                "status":"SENT",
                "created_ts": datetime.utcnow().isoformat(timespec="minutes"),
                "result":"Recorded (demo). Production would integrate to real systems."
            })
            audit("execution_recorded", f"approval={aid}")
            st.success("Executed (recorded).")

# ---------------------------------------------------------
# Audit
# ---------------------------------------------------------
with tabs[7]:
    st.subheader("Audit Log (demo)")
    st.write("Every action is recorded here (demo). Production: immutable audit + RBAC.")
    if st.session_state["audit"]:
        st.dataframe(pd.DataFrame(st.session_state["audit"]), use_container_width=True, height=380)
    else:
        st.info("No audit events yet. Run Autopilot or generate a copilot response.")

st.caption("⚠️ Synthetic demo only. A real NHS system needs formal IG, RBAC/SSO, audit controls, and integration to EPR/FHIR/PACS.")
