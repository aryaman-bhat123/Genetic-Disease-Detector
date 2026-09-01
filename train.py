import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


# ---------------------------------------------------------
# GENETIC VARIANT ML TRAINING
# ---------------------------------------------------------

DATA_PATH = Path("data/demo_variants.csv")
MODEL_PATH = Path("models/variant_classifier.joblib")

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# -------------------- LOAD DATA --------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded")
print("-" * 50)

print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(df.columns) - 1}")

# -------------------- TARGET --------------------

TARGET = "higher_risk_pattern"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

# -------------------- FEATURES --------------------

categorical_features = [
    "gene",
    "consequence",
    "variant_type",
    "family_history"
]

numeric_features = [
    "allele_frequency",
    "conservation",
    "functional_score"
]

# -------------------- PREPROCESSING --------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)

# -------------------- RANDOM FOREST --------------------

classifier = RandomForestClassifier(
    n_estimators=350,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# -------------------- COMPLETE PIPELINE --------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)

# -------------------- TRAIN / TEST SPLIT --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining model...")
print("-" * 50)

pipeline.fit(
    X_train,
    y_train
)

# -------------------- PREDICTIONS --------------------

predictions = pipeline.predict(
    X_test
)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

# -------------------- EVALUATION --------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("\nMODEL PERFORMANCE")
print("=" * 50)

print(
    f"Accuracy: {accuracy:.3f}"
)

print(
    f"ROC-AUC: {roc_auc:.3f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        digits=3
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# -------------------- SAVE MODEL --------------------

model_bundle = {
    "model": pipeline,

    "features": [
        "gene",
        "consequence",
        "variant_type",
        "allele_frequency",
        "conservation",
        "functional_score",
        "family_history"
    ],

    "target": TARGET,

    "model_type": "Random Forest",

    "training_note":
        "Synthetic educational dataset. "
        "Not clinically validated."
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nModel saved:")
print(
    MODEL_PATH
)

print("\nTraining complete.")
