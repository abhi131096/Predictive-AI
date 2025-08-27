import os                           # stdlib, for paths
import re                           # stdlib, for regex cleanup
import joblib                       # for saving models
import numpy as np                  # numerical arrays
import pandas as pd                 # data wrangling
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # CV tools
from sklearn.metrics import classification_report, f1_score                              # metrics
from sklearn.pipeline import Pipeline                                                     # pipeline
from sklearn.compose import ColumnTransformer                                             # feature union
from sklearn.feature_extraction.text import TfidfVectorizer                              # text vectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler                          # encoders
from sklearn.linear_model import SGDClassifier                                           # fast linear clf
from sklearn.semi_supervised import SelfTrainingClassifier                               # self training
from scipy import sparse                                                                  # to hstack safely
import warnings                                                                           # silence noisy warnings

# ------------- CONFIG -------------
RANDOM_STATE = 42                                   # fixed seed
ARTIFACT_DIR = "artifacts"                          # output directory
os.makedirs(ARTIFACT_DIR, exist_ok=True)            # create if missing

# You can change this to your CSV path
CSV_PATH = "interview_task_dataset.csv"             # expected CSV name

# Column names I will use, normalize headers to safe names first
TEXT_COL_RAW = "Time Narrative"                     # text column original header
CAT_COLS_RAW = ["Grade", "Charged to Client?"]      # categorical original headers
NUM_COLS_RAW = ["Worked Hrs"]                       # numeric original headers
TARGET_RAW = "Category"                             # target header

# ------------- CLEAN HELPERS -------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase headers, strip, replace spaces and slashes with underscores to avoid typos
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                   .str.replace(r"\s+", "_", regex=True)
                   .str.replace(r"[^\w]+", "_", regex=True)
                   .str.lower()
    )
    return df

def rename_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Map raw headers to normalized ones we will reference throughout
    mapping = {
        TEXT_COL_RAW: "text",
        "Time_Narrative": "text",
        "time_narrative": "text",
        CAT_COLS_RAW[0]: "grade",
        "grade": "grade",
        CAT_COLS_RAW[1]: "charged",
        "charged_to_client": "charged",
        "charged_to_client_": "charged",
        NUM_COLS_RAW[0]: "worked_hrs",
        "worked_time": "worked_hrs",
        "worked_hrs": "worked_hrs",
        TARGET_RAW: "category",
        "Category": "category",
        "category": "category",
        "Department": "department",   # optional in file, ignored by model
        "department": "department",
        "Record_ID": "record_id",
        "record_id": "record_id",
    }
    # Only rename keys present to avoid KeyError
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=present)

def clean_text_series(s: pd.Series) -> pd.Series:
    # Ensure string, lower, collapse whitespace, remove stray control chars
    s = s.astype(str)                                                        # cast to str for vectorizer
    s = s.str.lower()                                                        # lowercase
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()                   # collapse whitespace
    # common legal abbreviations normalization to help features
    replace_pairs = {
        "o/s": "other side",
        "os": "other side",
        "emo": "email out",
        "emi": "email in",
        "gp": "general practitioner",
        "pfdr": "private fdr",
        "fdr": "financial dispute resolution",
        "fhdra": "first hearing dispute resolution appointment",
        "loe": "letter of engagement",
        "ra": "risk assessment",
        "es1": "case summary",
        "es2": "schedule of assets",
        "d81": "d eighty one",
        "form e": "form e",
        "cafcass": "cafcass",
    }
    # simple token replacements
    for k, v in replace_pairs.items():
        s = s.str.replace(rf"\b{k}\b", v, regex=True)
    return s

# ------------- PREPROCESSOR -------------

def make_preprocessor(text_col: str, cat_cols: list, num_cols: list) -> ColumnTransformer:
    # Tfidf for text, OHE for categoricals, scaling for numerics
    text_pipe = TfidfVectorizer(
        ngram_range=(1,2),             # unigrams and bigrams
        min_df=2,                      # ignore rare noise
        max_features=30000,            # cap vocabulary to avoid RAM blowup
        dtype=np.float32               # smaller memory footprint
    )
    # OneHotEncoder API changed, use 'sparse_output' for sklearn >=1.2
    # We want a sparse matrix to stack with tfidf
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    scaler = StandardScaler(with_mean=False)        # with_mean=False to keep sparse compatibility

    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, text_col),          # text to tfidf
            ("cat", ohe, cat_cols),                 # categoricals to OHE sparse
            ("num", scaler, num_cols),              # numeric to scaled sparse
        ],
        remainder="drop",
        sparse_threshold=1.0                        # force sparse output
    )
    return pre

# ------------- MODELS -------------

def make_supervised_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    # Linear classifier that converges fast on sparse high dim TFIDF
    clf = SGDClassifier(
        loss="log_loss",                # probabilistic output for self-training threshold
        alpha=1e-4,                     # regularization
        max_iter=2000,                  # increase to avoid ConvergenceWarning
        tol=1e-4,
        class_weight="balanced",        # handle label imbalance
        random_state=RANDOM_STATE
    )
    pipe = Pipeline([
        ("pre", preprocessor),          # first transform features
        ("clf", clf)                    # then fit classifier
    ])
    return pipe

def make_self_training(base_pipe: Pipeline, threshold: float=0.9) -> SelfTrainingClassifier:
    # Wrap only the classifier with SelfTraining, NOT raw text DataFrame
    # We will manually transform X via preprocessor and pass numpy arrays to SelfTraining
    base_clf = base_pipe.named_steps["clf"]                 # extract estimator
    self_train = SelfTrainingClassifier(
        estimator=base_clf,                                  # new param name in 1.6+
        threshold=threshold,                                 # only high confidence
        verbose=True                                         # print progress
    )
    return self_train

# ------------- TRAINING -------------

def train_supervised(df_lab: pd.DataFrame, pre: ColumnTransformer) -> Pipeline:
    # Build pipeline and run CV, then fit on all labelled
    pipe = make_supervised_pipeline(pre)                                              # create pipeline
    X = df_lab[["text","grade","charged","worked_hrs"]]                                # features
    y = df_lab["category"]                                                             # labels

    # 5 fold stratified CV on macro F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)          # CV splitter
    f1_macro = cross_val_score(pipe, X, y, scoring="f1_macro", cv=cv, n_jobs=1)        # n_jobs=1 avoids loky crashes on Windows

    print(f"Supervised CV macro F1: {f1_macro.mean():.3f} ± {f1_macro.std():.3f}")     # quick score

    pipe.fit(X, y)                                                                      # fit on full labelled
    return pipe

def train_self_training(df_lab: pd.DataFrame, df_unlab: pd.DataFrame,
                        pre: ColumnTransformer, sup_pipe: Pipeline,
                        threshold: float=0.9) -> SelfTrainingClassifier:
    # Transform X once with the preprocessor to numeric sparse matrices
    X_lab = pre.transform(df_lab[["text","grade","charged","worked_hrs"]])              # sparse features labelled
    X_un = pre.transform(df_unlab[["text","grade","charged","worked_hrs"]])             # sparse features unlabelled

    # Vertically stack to a single sparse matrix
    X_all = sparse.vstack([X_lab, X_un], format="csr")                                  # combine features
    # y: labelled actual classes followed by -1 for unlabelled
    y_all = np.concatenate([df_lab["category"].astype(str).values,
                            np.full(len(df_unlab), -1, dtype=object)])                  # -1 sentinel

    # Build SelfTraining on top of the ALREADY FIT supervised classifier's estimator
    self_train = make_self_training(sup_pipe, threshold=threshold)                      # wrapper

    # Fit on numeric arrays, not on raw strings, avoids "could not convert string to float"
    self_train.fit(X_all, y_all)                                                        # semi-supervised fit
    return self_train

# ------------- MAIN -------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning)        # keep output tidy
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Load CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8", engine="python")  # robust read

    # Normalize and rename headers to expected names
    df = normalize_columns(df)                                     # safe headers
    df = rename_expected_columns(df)                               # align names

    # Validate required columns, give clear error if missing
    required = {"text","grade","charged","worked_hrs","category"}
    missing = [c for c in required if c not in df.columns]         # check presence
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Present columns: {list(df.columns)}")

    # Clean text
    df["text"] = clean_text_series(df["text"])                     # clean narrative

    # Fix dtypes for categoricals and numeric
    df["grade"] = df["grade"].astype(str).str.strip().replace({"nan":"Unknown"})       # categorical as str
    df["charged"] = df["charged"].astype(str).str.strip().str.upper().map(             # YES NO unify
        {"YES":"YES","NO":"NO"}).fillna("NO")
    # Convert worked hours, coerce bad to NaN then fill with median
    df["worked_hrs"] = pd.to_numeric(df["worked_hrs"], errors="coerce")                # numeric cast
    df["worked_hrs"] = df["worked_hrs"].fillna(df["worked_hrs"].median())              # fill

    # Split labelled vs unlabelled
    labelled = df[df["category"].notna()].copy()                   # rows with labels
    unlabelled = df[df["category"].isna()].copy()                  # rows without labels

    # Build preprocessor using only columns that exist
    text_col = "text"                                              # set text name
    cat_cols = ["grade","charged"]                                 # two categoricals
    num_cols = ["worked_hrs"]                                      # numeric

    pre = make_preprocessor(text_col, cat_cols, num_cols)          # create transformer
    pre.fit(labelled[[text_col]+cat_cols+num_cols])                # fit vocab and encoders only on labelled to avoid target leakage

    # Train supervised pipeline
    sup_pipe = train_supervised(labelled, pre)                     # returns fitted pipeline

    # Evaluate simple holdout for quick human-readable metrics
    X_tr, X_te, y_tr, y_te = train_test_split(
        labelled[["text","grade","charged","worked_hrs"]],
        labelled["category"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labelled["category"]
    )
    # Fit and predict
    sup_pipe.fit(X_tr, y_tr)                                       # fit on train
    y_pred = sup_pipe.predict(X_te)                                 # predict on test
    print("\nSupervised only report")
    print(classification_report(y_te, y_pred, digits=2))            # readable report

    # Self-training only if we actually have unlabelled rows
    if len(unlabelled) > 0:
        # IMPORTANT: transform to numeric before SelfTraining to avoid string-to-float error
        self_train_model = train_self_training(labelled, unlabelled, pre, sup_pipe, threshold=0.9)
        # After self-training, refit full supervised pipeline on pseudo-labelled data for a final model
        # Build combined dataset with pseudo labels
        X_un = unlabelled[["text","grade","charged","worked_hrs"]]
        X_lab = labelled[["text","grade","charged","worked_hrs"]]
        # Predict high-confidence pseudo labels using self-training estimator
        # We must call decision_function or predict_proba on transformed X
        X_un_t = pre.transform(X_un)
        pseudo = self_train_model.predict(X_un_t)
        # Merge
        df_all = pd.concat([labelled, unlabelled.copy()], axis=0, ignore_index=True)
        df_all.loc[df_all.index >= len(labelled), "category"] = pseudo
        # Final train on all
        final_pipe = make_supervised_pipeline(pre)
        final_pipe.fit(df_all[["text","grade","charged","worked_hrs"]], df_all["category"])
    else:
        final_pipe = sup_pipe

    # Save artefacts
    joblib.dump(pre, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))                # save transformer
    joblib.dump(final_pipe, os.path.join(ARTIFACT_DIR, "model_pipeline.joblib"))       # save final pipeline

    # Predict categories for any unlabelled and write CSV for review
    if len(unlabelled) > 0:
        preds = final_pipe.predict(unlabelled[["text","grade","charged","worked_hrs"]])
        out = unlabelled.copy()
        out["predicted_category"] = preds
        out.to_csv(os.path.join(ARTIFACT_DIR, "unlabelled_with_predictions.csv"),
                   index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
