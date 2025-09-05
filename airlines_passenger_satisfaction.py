import streamlit as st
import pandas as pd
import joblib

# ---- Konfigurasi kolom dataset ----
NOMINAL_COLS = ["Gender", "Customer Type", "Type of Travel"]
ORDINAL_COLS = ["Class"]
RATING_COLS = [
    "Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking",
    "Gate location","Food and drink","Online boarding","Seat comfort",
    "Inflight entertainment","On-board service","Leg room service","Baggage handling",
    "Checkin service","Inflight service","Cleanliness"
]
NUMERIC_COLS = ["Age","Flight Distance","Departure Delay in Minutes","Arrival Delay in Minutes"]
ALL_FEATURES = NOMINAL_COLS + ORDINAL_COLS + RATING_COLS + NUMERIC_COLS
LABEL_MAP = {0: "Satisfied", 1: "Neutral or Dissatisfied"}

st.set_page_config(page_title="Airline Satisfaction", page_icon="✈️", layout="centered")
st.title("✈️ Airlines Passenger Satisfaction Prediction")

# ---- Load model dari folder models/ ----
@st.cache_resource
def load_model(path="models/satisfaction_pipeline.pkl"):
    return joblib.load(path)

with st.spinner("Memuat model…"):
    clf = load_model()
st.success("Model siap ✅")

# ---- Helper ----
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in ALL_FEATURES:
        if c not in df2.columns:
            if c in RATING_COLS: df2[c] = 3
            elif c in NOMINAL_COLS: df2[c] = ""
            elif c in ORDINAL_COLS: df2[c] = "Eco"
            elif c in NUMERIC_COLS: df2[c] = 0
    return df2[ALL_FEATURES]

def predict_df(df_in: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    X = ensure_columns(df_in)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = X.copy()
    out["pred_proba_dissatisfied"] = proba
    out["prediction"] = [LABEL_MAP[p] for p in pred]
    return out

# ---- UI ----
st.subheader("Pengaturan Prediksi")
thresh = st.slider("Ambang (threshold) untuk kelas 'Neutral or Dissatisfied'",
                   0.05, 0.95, 0.50, 0.05)

st.subheader("Prediksi Satu Penumpang (Form)")
with st.form("single_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female","Male"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer","disloyal customer"])
        travel_type = st.selectbox("Type of Travel", ["Personal Travel","Business Travel"])
        travel_class = st.selectbox("Class", ["Eco","Eco Plus","Business"])
        age = st.number_input("Age", 0, 120, 35)
        flight_distance = st.number_input("Flight Distance", 0, 20000, 800)
    with c2:
        dep_delay = st.number_input("Departure Delay in Minutes", 0, 3000, 0)
        arr_delay = st.number_input("Arrival Delay in Minutes", 0, 3000, 0)
        ratings = {}
        for i, col in enumerate(RATING_COLS):
            ratings[col] = st.slider(col, 0, 5, 3, key=f"r{i}")

    if st.form_submit_button("Prediksi", use_container_width=True):
        row = {
            "Gender": gender, "Customer Type": customer_type, "Type of Travel": travel_type,
            "Class": travel_class, "Age": age, "Flight Distance": flight_distance,
            "Departure Delay in Minutes": dep_delay, "Arrival Delay in Minutes": arr_delay,
            **ratings
        }
        res = predict_df(pd.DataFrame([row]), threshold=thresh)
        label = res.loc[0, "prediction"]
        p = float(res.loc[0, "pred_proba_dissatisfied"])
        msg = f"Prediksi: **{label}** (Prob. dissatisfied: {p:.2f}, threshold={thresh:.2f})"

        if label == "Neutral or Dissatisfied":
            st.error(msg, icon="❌")
        else:
            st.success(msg, icon="✅")

st.subheader("Batch Prediction (Upload CSV)")
up = st.file_uploader("Upload CSV (tanpa kolom 'satisfaction')", type=["csv"])
if up is not None:
    df_new = pd.read_csv(up)
    res = predict_df(df_new, threshold=thresh)
    st.dataframe(res[["prediction","pred_proba_dissatisfied"] + ALL_FEATURES],
                 use_container_width=True)
