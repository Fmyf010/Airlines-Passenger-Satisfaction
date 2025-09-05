from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import streamlit as st

# ------------------ Path ------------------
PKL_PATH = Path("satisfaction_pipeline.pkl")      # simpan model di root repo
DATA_PATH = Path("Airlines Passanger.csv")        # CSV di root repo
TARGET = "satisfaction"

# ------------------ Training jika model belum ada ------------------
if not PKL_PATH.exists():
    st.write("🔄 Melatih model cepat karena .pkl belum ada ...")

    NOMINAL_COLS = ["Gender", "Customer Type", "Type of Travel"]
    ORDINAL_COLS = ["Class"]
    RATING_COLS  = [
        "Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking",
        "Gate location","Food and drink","Online boarding","Seat comfort",
        "Inflight entertainment","On-board service","Leg room service","Baggage handling",
        "Checkin service","Inflight service","Cleanliness"
    ]
    NUMERIC_COLS = ["Age","Flight Distance","Departure Delay in Minutes","Arrival Delay in Minutes"]

    df = pd.read_csv(DATA_PATH)
    for c in ["Unnamed: 0","id"]:
        if c in df.columns: 
            df = df.drop(columns=c)

    assert TARGET in df.columns, f"Kolom target '{TARGET}' tidak ada di CSV."
    X = df.drop(columns=TARGET)
    y = df[TARGET].map({"neutral or dissatisfied": 1, "satisfied": 0})

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)

    # Preprocessor
    nominal_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    ordinal_pipe = Pipeline([("ord", OrdinalEncoder(categories=[["Eco","Eco Plus","Business"]]))])
    rating_pipe  = Pipeline([("scale", MinMaxScaler())])
    dep_pipe = Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                         ("scale", MinMaxScaler())])
    arr_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                         ("scale", MinMaxScaler())])
    age_pipe, dist_pipe = MinMaxScaler(), MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipe, NOMINAL_COLS),
            ("ordinal", ordinal_pipe, ORDINAL_COLS),
            ("rating",  rating_pipe,  RATING_COLS),
            ("age",     age_pipe,     ["Age"]),
            ("dist",    dist_pipe,    ["Flight Distance"]),
            ("dep",     dep_pipe,     ["Departure Delay in Minutes"]),
            ("arr",     arr_pipe,     ["Arrival Delay in Minutes"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    xgb = XGBClassifier(
        random_state=42,
        n_estimators=150,
        learning_rate=0.10,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
    )

    clf = Pipeline([("prep", preprocessor), ("model", xgb)])
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    st.text(classification_report(yte, pred, zero_division=0))
    st.text(f"AUC: {roc_auc_score(yte, proba):.3f}")

    joblib.dump(clf, PKL_PATH.as_posix())
    st.success(f"✅ Model dilatih dan disimpan: {PKL_PATH}")

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Airline Satisfaction", page_icon="✈️", layout="centered")
st.title("✈️ Airlines Passenger Satisfaction Prediction")

@st.cache_resource
def load_model(path=PKL_PATH):
    return joblib.load(path)

clf = load_model()

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
LABEL_MAP = {0:"Satisfied", 1:"Neutral or Dissatisfied"}

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
    pred  = (proba >= threshold).astype(int)
    out = X.copy()
    out["pred_proba_dissatisfied"] = proba
    out["prediction"] = [LABEL_MAP[p] for p in pred]
    return out

# ---- UI ----
st.subheader("Prediction Setting")
thresh = st.slider("Threshold for class 'Neutral or Dissatisfied'", 0.05, 0.95, 0.50, 0.05)

st.subheader("Passenger Prediction (for 1 Passenger)")
with st.form("single_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female","Male"], key="g")
        customer_type = st.selectbox("Customer Type", ["Loyal Customer","disloyal customer"], key="ct")
        travel_type = st.selectbox("Type of Travel", ["Personal Travel","Business Travel"], key="tt")
        travel_class = st.selectbox("Class", ["Eco","Eco Plus","Business"], key="cl")
        age = st.number_input("Age", 0, 120, 35, key="age")
        flight_distance = st.number_input("Flight Distance", 0, 20000, 800, key="fd")
    with c2:
        dep_delay = st.number_input("Departure Delay in Minutes", 0, 3000, 0, key="dep")
        arr_delay = st.number_input("Arrival Delay in Minutes", 0, 3000, 0, key="arr")
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
        p     = float(res.loc[0, "pred_proba_dissatisfied"])
        msg   = f"Prediksi: **{label}** (Prob. dissatisfied: {p:.2f}, threshold={thresh:.2f})"

        if label == "Neutral or Dissatisfied":
            st.error(msg, icon="❌")
        else:
            st.success(msg, icon="✅")

st.subheader("Batch Prediction (Upload CSV)")
up = st.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    df_new = pd.read_csv(up)
    res = predict_df(df_new, threshold=thresh)
    st.dataframe(res[["prediction","pred_proba_dissatisfied"] + ALL_FEATURES], use_container_width=True)
