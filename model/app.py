from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ------------------------------------------------------------
# Page setup and visual styling
# ------------------------------------------------------------
st.set_page_config(page_title="CerviCare | ML Demo", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {--primary-color:#168178; --cc-ink:#173047; --cc-muted:#647887; --cc-line:#dce7e4; --cc-canvas:#f4f7f6;}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:var(--cc-canvas); color:var(--cc-ink);}
.block-container {padding-top:2.25rem; padding-bottom:2rem; max-width:1420px;}
[data-testid="stSidebar"] {background:#153b40; min-width:250px;}
[data-testid="stSidebar"] * {color:#eef5ff !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:7px 10px; border-radius:8px;}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {background:rgba(97,203,177,.14); border-radius:8px;}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {border-color:#61cbb1 !important; background-color:#61cbb1 !important;}
.hero {background:linear-gradient(112deg,#153d4a 0%,#14665f 100%); padding:20px 24px; border-radius:8px; color:white; margin-bottom:12px;}
.hero h1 {font-family:'Manrope',sans-serif; font-size:29px; color:white; margin:0 0 4px 0; font-weight:800;}
.hero p {color:#d8eaf4; margin:0; font-size:15px;}
.eyebrow {font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#a8d8db; margin-bottom:6px;}
.section-title {font-family:'Manrope',sans-serif; font-size:20px; font-weight:800; color:#173657; margin:12px 0 4px 0;}
.subtle {color:var(--cc-muted); font-size:13px;}
.metric-card {background:white; border:1px solid var(--cc-line); border-radius:8px; padding:13px 16px; min-height:96px;}
.metric-label {color:#617887; font-size:12px; font-weight:600; margin-bottom:7px;}
.metric-value {font-family:'Manrope',sans-serif; color:#173f48; font-size:25px; font-weight:800; line-height:1.1;}
.metric-foot {font-size:11px; color:#738792; margin-top:6px;}
.warning {background:#fffaf0; border:1px solid #efe1c4; border-left:4px solid #bd7a1e; padding:9px 12px; border-radius:8px; color:#654c25; font-size:12px; margin-bottom:14px;}
.stButton>button[kind="primary"] {background:#168178; border:0; border-radius:8px; font-weight:700; padding:10px 18px;}
.stButton>button[kind="primary"]:hover {background:#11665f;}
div[data-testid="stForm"] {background:white; border:1px solid var(--cc-line); border-radius:8px; padding:16px;}
@media (max-width: 720px) {
    .block-container {padding:0.75rem 0.8rem 1.5rem;}
    .hero {padding:17px 16px;}
    .hero h1 {font-size:25px;}
    .metric-card {padding:12px; min-height:88px;}
    .metric-value {font-size:22px;}
}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cervical-cancer_csv (1).csv"

# Same preprocessing, predictors, split, and estimator as the supplied notebook.
@st.cache_data(show_spinner=False)
def load_data(path_string, modified_time):
    df = pd.read_csv(path_string)
    df = df.fillna(0)
    df = df.drop(columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis", "Unnamed: 0"], errors="ignore")
    return df

@st.cache_resource(show_spinner="Training the notebook's Logistic Regression model...")
def train_model(data):
    X = data.drop(columns=["Biopsy"])
    y = data["Biopsy"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X.columns.tolist(), y_test, y_pred

# Human-readable groupings. All feature names remain exactly those used by the notebook.
GROUPS = {
    "General information": ["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies"],
    "Lifestyle": ["Smokes", "Smokes (years)", "Smokes (packs/year)"],
    "Contraceptive history": ["Hormonal Contraceptives", "Hormonal Contraceptives (years)", "IUD", "IUD (years)"],
    "STI-related variables": ["STDs", "STDs (number)", "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV", "STDs: Number of diagnosis"],
    "Previous diagnoses & screening indicators": ["Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx", "Hinselmann", "Schiller", "Citology"]
}
BINARY = {"Smokes", "Hormonal Contraceptives", "IUD", "STDs", "STDs:condylomatosis", "STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx", "Hinselmann", "Schiller", "Citology"}
DISPLAY = {
    "Age":"Age", "Number of sexual partners":"Number of sexual partners", "First sexual intercourse":"Age at first sexual intercourse", "Num of pregnancies":"Number of pregnancies",
    "Smokes":"Smoking history", "Smokes (years)":"Years of smoking", "Smokes (packs/year)":"Smoking exposure (packs/year)",
    "Hormonal Contraceptives":"Hormonal contraceptive history", "Hormonal Contraceptives (years)":"Years using hormonal contraceptives", "IUD":"IUD history", "IUD (years)":"Years with IUD",
    "STDs":"STI history recorded", "STDs (number)":"Number of STIs recorded", "STDs: Number of diagnosis":"Number of STI diagnoses",
    "Dx:Cancer":"Previous cancer diagnosis indicator", "Dx:CIN":"Previous CIN diagnosis indicator", "Dx:HPV":"Previous HPV diagnosis indicator", "Dx":"Other previous diagnosis indicator",
    "Hinselmann":"Hinselmann test indicator", "Schiller":"Schiller test indicator", "Citology":"Cytology test indicator"
}

def pretty(name):
    return DISPLAY.get(name, name.replace("STDs:", "STI: "))

def metric(label, value, foot=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-size:23px;font-weight:800;margin:6px 0 0'>🩺 CerviCare</div><div style='font-size:11px;color:#b7cadc;margin:3px 0 22px'>CERVICAL HEALTH · ML PROJECT</div>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Overview", "Prediction demo", "Model performance", "Dataset explorer"], label_visibility="visible")
    st.markdown("<hr style='border-color:#34516f;margin:20px 0'>", unsafe_allow_html=True)
    st.caption("Academic machine-learning prototype")
    st.caption("Model: Logistic Regression")

PAGE_COPY = {
    "Overview": ("CerviCare overview", "Risk-factor data and model setup at a glance."),
    "Prediction demo": ("Prediction demo", "Explore model outputs using the dataset's numeric encodings."),
    "Model performance": ("Model performance", "Review held-out metrics beside a majority-class baseline."),
    "Dataset explorer": ("Dataset explorer", "Inspect the source data and processed feature summary."),
}
hero_title, hero_description = PAGE_COPY[page]
st.markdown(f"<div class='hero'><div class='eyebrow'>CerviCare · Machine learning project</div><h1>{hero_title}</h1><p>{hero_description}</p></div>", unsafe_allow_html=True)
st.markdown("<div class='warning'><b>Educational use only.</b> Not a medical device or clinical risk assessment. Do not use these outputs for care decisions. Consult a qualified healthcare professional.</div>", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}\n\nKeep your existing folder structure and ensure the CSV is inside the data folder with the name `cervical-cancer_csv (1).csv`.")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model, features, y_test, y_pred = train_model(data)
except Exception as exc:
    st.error(f"Could not load the dataset or train the model: {exc}")
    st.stop()

if page == "Overview":
    st.markdown("<div class='section-title'>Project at a glance</div><div class='subtle'>A quick summary of the data and the model setup.</div>", unsafe_allow_html=True)
    total, feature_count = len(data), len(features)
    positive = int(data["Biopsy"].sum())
    records_col, features_col = st.columns(2)
    with records_col: metric("Dataset records", f"{total:,}", "Rows available after notebook preprocessing")
    with features_col: metric("Input features", str(feature_count), "Predictors used by the notebook")
    positives_col, split_col = st.columns(2)
    with positives_col: metric("Target-positive rows", f"{positive:,}", "Biopsy target = 1 in this dataset")
    with split_col: metric("Test split", "20%", "Stratified split · random state 42")
    st.write("")
    left,right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Workflow")
        st.markdown("""
        1. Load the CSV from the existing `data` folder.
        2. Fill missing values with `0`, as in the notebook.
        3. Drop the two STI diagnosis-time columns and any `Unnamed: 0` index column.
        4. Split predictors and the `Biopsy` target using a stratified 80/20 split.
        5. Train a pipeline with median imputation, standard scaling, and balanced Logistic Regression.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Target distribution")
        counts = data["Biopsy"].value_counts().reindex([0,1], fill_value=0)
        st.bar_chart(pd.DataFrame({"Records":counts.values}, index=["Biopsy = 0", "Biopsy = 1"]), color="#27818a")
        st.caption("Counts reflect the dataset labels, not population prevalence.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Prediction demo":
    st.markdown("<div class='section-title'>Try the model</div><div class='subtle'>Enter values using the dataset's numeric coding. No information is saved by this app.</div>", unsafe_allow_html=True)
    st.info("This form is a software demonstration. It is not intended for entering real patient details or interpreting a person's health.")
    with st.form("demo_form"):
        entered = {}
        for group, names in GROUPS.items():
            names = [n for n in names if n in features]
            if not names: continue
            with st.expander(group, expanded=(group == "General information")):
                cols = st.columns(2)
                for i, name in enumerate(names):
                    s = pd.to_numeric(data[name], errors="coerce").dropna()
                    lo = float(s.min()) if len(s) else 0.0
                    hi = float(s.max()) if len(s) else 1.0
                    default = float(s.median()) if len(s) else 0.0
                    with cols[i % 2]:
                        if name in BINARY:
                            val = st.selectbox(pretty(name), [0,1], index=int(round(default)) if default in (0,1) else 0, format_func=lambda x: "No / 0" if x == 0 else "Yes / 1", help="Numeric encoding used in the dataset.")
                        else:
                            step = 1.0 if float(lo).is_integer() and float(hi).is_integer() else 0.5
                            val = st.number_input(pretty(name), min_value=float(lo), max_value=float(hi), value=float(np.clip(default,lo,hi)), step=step, help=f"Observed dataset range: {lo:g} to {hi:g}. This is a dataset range, not a recommended clinical range.")
                        entered[name] = float(val)
        submitted = st.form_submit_button("Run demo prediction", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{f: entered.get(f, float(pd.to_numeric(data[f], errors="coerce").median() or 0)) for f in features}], columns=features)
        pred = int(model.predict(row)[0])
        probs = model.predict_proba(row)[0]
        st.markdown("### Model output")
        a,b = st.columns(2)
        with a: metric("Predicted dataset class", str(pred), "Class label only · not a diagnosis")
        with b: metric("Model score for class 1", f"{probs[1]:.1%}", "Model output; not an individual's clinical probability")
        st.progress(float(probs[1]))
        st.caption("The model score reflects this fitted model and its dataset coding. It is not calibrated or clinically validated.")

elif page == "Model performance":
    st.markdown("<div class='section-title'>Model performance</div><div class='subtle'>Metrics on the held-out test portion generated with the notebook's split settings.</div>", unsafe_allow_html=True)
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred,zero_division=0)
    rec = recall_score(y_test,y_pred,zero_division=0)
    f1 = f1_score(y_test,y_pred,zero_division=0)
    accuracy_col, precision_col = st.columns(2)
    with accuracy_col: metric("Accuracy",f"{acc:.1%}","Overall correct classifications")
    with precision_col: metric("Precision · class 1",f"{prec:.1%}","Positive predictions that were correct")
    recall_col, f1_col = st.columns(2)
    with recall_col: metric("Recall · class 1",f"{rec:.1%}","Actual class 1 cases detected")
    with f1_col: metric("F1 · class 1",f"{f1:.1%}","Harmonic mean of precision and recall")
    left,right = st.columns([.8,1.2], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test,y_pred,labels=[0,1])
        st.dataframe(pd.DataFrame(cm,index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]),width="stretch")
        st.caption("Counts for the held-out test split.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Classification report")
        report = classification_report(y_test,y_pred,output_dict=True,zero_division=0)
        rep = pd.DataFrame(report).T
        st.dataframe(rep.round(3),width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Metrics are dataset-specific and do not establish clinical safety, validity, or real-world performance.")

elif page == "Dataset explorer":
    st.markdown("<div class='section-title'>Explore the dataset</div><div class='subtle'>A lightweight view of the CSV after the notebook's preprocessing steps.</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Preview")
        st.dataframe(data.head(12),width="stretch", height=360)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Feature summary")
        summary = data.describe().T[["mean","std","min","max"]].round(2)
        st.dataframe(summary,width="stretch",height=360)
        st.markdown("</div>", unsafe_allow_html=True)
    st.download_button("Download processed dataset", data.to_csv(index=False).encode("utf-8"), file_name="cervical_dataset_processed.csv", mime="text/csv")

st.markdown("<div style='text-align:center;color:#8a98aa;font-size:11px;margin-top:35px'>CerviCare · Academic ML interface · Not for clinical use</div>", unsafe_allow_html=True)
