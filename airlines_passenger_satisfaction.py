from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

PKL_PATH = Path("/content/satisfaction_pipeline.pkl")
DATA_PATH = "/data/Airlines Passanger.csv"
TARGET = "satisfaction"

if PKL_PATH.exists():
    print("✅ Model sudah ada:", PKL_PATH)
else:
    print("Melatih model cepat dan menyimpan ke .pkl …")

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
        if c in df.columns: df = df.drop(columns=c)

    assert TARGET in df.columns, f"Kolom target '{TARGET}' tidak ada di CSV."
    X = df.drop(columns=TARGET)
    y = df[TARGET].map({"neutral or dissatisfied": 1, "satisfied": 0})

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)

    # Preprocessor → semua ke numerik
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
        n_estimators=150,      # cepat di CPU Colab
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

    # Eval ringkas (opsional)
    proba = clf.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    print(classification_report(yte, pred, zero_division=0))
    print("AUC:", roc_auc_score(yte, proba))

    joblib.dump(clf, PKL_PATH.as_posix())
    print("Disimpan:", PKL_PATH)
