import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Cervical Health Risk Demo", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {background: #f6f8fc;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.4rem 1.7rem; border-radius: 18px; background: linear-gradient(110deg,#173b67,#286c85); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0;}
.note {background:#fff4df; padding:1rem; border-left:5px solid #e6a23c; border-radius:8px; color:#654b1b;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🩺 Cervical Health Risk Analysis</h1><p>Machine-learning demonstration based on the notebook’s Logistic Regression workflow</p></div>', unsafe_allow_html=True)
st.markdown('<div class="note"><b>Important:</b> This educational prototype is not a medical test or diagnosis. Do not use its output to make healthcare decisions. Consult a qualified healthcare professional for screening and medical advice.</div>', unsafe_allow_html=True)

@st.cache_resource
def train_model(file_bytes):
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes))
    df = df.drop(columns=[
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis",
        "Unnamed: 0"
    ], errors="ignore")
    if "Biopsy" not in df.columns:
        raise ValueError("The uploaded CSV must contain a 'Biopsy' target column.")
    df = df.replace("?", np.nan)
    y = pd.to_numeric(df["Biopsy"], errors="coerce")
    X = df.drop(columns=["Biopsy"]).apply(pd.to_numeric, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, X.columns.tolist(), y_test, pred, len(df)

with st.sidebar:
    st.header("Data & model")
    uploaded = st.file_uploader("Upload cervical-cancer_csv.csv", type=["csv"])
    st.caption("The CSV is used locally by this Streamlit session to train the model.")
    st.divider()
    st.caption("Model: median imputation → standard scaling → balanced Logistic Regression")

if uploaded is None:
    st.info("Upload the same CSV dataset used in your Jupyter notebook to activate the prediction form.")
    st.markdown("### What this app includes")
    st.write("- Patient-feature input form generated from the dataset columns")
    st.write("- Model output and probability display")
    st.write("- Basic held-out test-set metrics")
    st.stop()

try:
    model, features, y_test, pred, n_rows = train_model(uploaded.getvalue())
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

st.success(f"Model trained using {n_rows:,} dataset rows.")
tab1, tab2 = st.tabs(["Prediction demo", "Model evaluation"])

with tab1:
    st.subheader("Enter feature values")
    st.caption("Fields are generated from the notebook's predictor columns. Use values consistent with the dataset's coding.")
    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {}
        for i, feature in enumerate(features):
            with cols[i % 3]:
                label = feature.replace("_", " ")
                # Binary variables are displayed as yes/no where their observed values are 0 and 1.
                series = pd.to_numeric(pd.Series([], dtype=float), errors="coerce")
                # The common dataset uses numeric encodings; safe generic inputs avoid assuming clinical ranges.
                values[feature] = st.number_input(label, value=0.0, step=1.0, format="%.2f", help="Enter a numeric value using the dataset's encoding.")
        submitted = st.form_submit_button("Run demo prediction", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([values], columns=features)
        probability = float(model.predict_proba(row)[0][1])
        prediction = int(model.predict(row)[0])
        st.divider()
        st.subheader("Model output")
        a, b = st.columns(2)
        a.metric("Predicted class", str(prediction))
        b.metric("Model-estimated probability for class 1", f"{probability:.1%}")
        st.caption("This is a model score for the dataset target, not an individual's clinical risk estimate.")
        st.progress(probability)

with tab2:
    st.subheader("Held-out test-set performance")
    st.caption("Evaluation on a random 20% split, matching the notebook's test_size=0.2 and random_state=42 setup.")
    st.metric("Accuracy", f"{accuracy_score(y_test, pred):.1%}")
    st.text("Classification report")
    st.code(classification_report(y_test, pred, zero_division=0))
    st.text("Confusion matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, pred), index=["Actual 0","Actual 1"], columns=["Predicted 0","Predicted 1"]), use_container_width=True)

st.divider()
st.caption("Prototype for academic demonstration only • Not validated for clinical use")
