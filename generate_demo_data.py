import numpy as np
import pandas as pd

from pathlib import Path


# ---------------------------------------------------------
# SYNTHETIC GENETIC VARIANT DATASET
#
# IMPORTANT:
# This dataset is artificial and is ONLY for demonstrating
# the machine-learning software.
# ---------------------------------------------------------

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)

# -------------------- OPTIONS --------------------

genes = [
    "BRCA1",
    "BRCA2",
    "CFTR",
    "HBB",
    "LDLR",
    "TP53",
    "PAH",
    "FBN1",
    "MECP2",
    "DMD",
    "SCN1A",
    "MLH1"
]

consequences = [
    "missense",
    "nonsense",
    "frameshift",
    "splice_site",
    "synonymous"
]

variant_types = [
    "SNV",
    "insertion",
    "deletion"
]

family_history_options = [
    "unknown",
    "no",
    "yes"
]

# -------------------- GENERATE DATA --------------------

NUM_SAMPLES = 1600

rows = []

for _ in range(NUM_SAMPLES):

    gene = rng.choice(
        genes
    )

    consequence = rng.choice(
        consequences,
        p=[
            0.38,
            0.15,
            0.15,
            0.12,
            0.20
        ]
    )

    variant_type = rng.choice(
        variant_types,
        p=[
            0.82,
            0.09,
            0.09
        ]
    )

    allele_frequency = float(
        np.clip(
            10 ** rng.uniform(
                -6,
                -0.2
            ),
            0,
            1
        )
    )

    conservation = float(
        np.clip(
            rng.beta(
                5,
                2
            ),
            0,
            1
        )
    )

    functional_score = float(
        np.clip(
            rng.beta(
                3,
                2
            ),
            0,
            1
        )
    )

    family_history = rng.choice(
        family_history_options,
        p=[
            0.55,
            0.30,
            0.15
        ]
    )

    # -----------------------------------------------------
    # SYNTHETIC LABEL GENERATION
    #
    # This is NOT a biological truth model.
    # It only creates patterns for ML experimentation.
    # -----------------------------------------------------

    score = (

        1.4
        * (
            consequence
            in [
                "nonsense",
                "frameshift",
                "splice_site"
            ]
        )

        + 0.9
        * conservation

        + 1.0
        * functional_score

        + 0.6
        * (
            allele_frequency
            < 0.001
        )

        + 0.7
        * (
            family_history
            == "yes"
        )

        + 0.2
        * (
            variant_type
            != "SNV"
        )

        + rng.normal(
            0,
            0.55
        )
    )

    higher_risk_pattern = int(
        score > 2.15
    )

    rows.append({
        "gene": gene,

        "consequence": consequence,

        "variant_type": variant_type,

        "allele_frequency":
            allele_frequency,

        "conservation":
            conservation,

        "functional_score":
            functional_score,

        "family_history":
            family_history,

        "higher_risk_pattern":
            higher_risk_pattern
    })


# -------------------- DATAFRAME --------------------

df = pd.DataFrame(
    rows
)

# -------------------- SAVE --------------------

output_directory = Path(
    "data"
)

output_directory.mkdir(
    exist_ok=True
)

output_file = (
    output_directory
    / "demo_variants.csv"
)

df.to_csv(
    output_file,
    index=False
)

# -------------------- REPORT --------------------

print("=" * 60)
print("SYNTHETIC GENETIC VARIANT DATASET")
print("=" * 60)

print(
    f"Samples generated: {len(df)}"
)

print(
    f"Features: {len(df.columns) - 1}"
)

print(
    f"Higher-risk examples: "
    f"{df['higher_risk_pattern'].sum()}"
)

print(
    f"Lower-risk examples: "
    f"{(df['higher_risk_pattern'] == 0).sum()}"
)

print(
    f"\nSaved to: {output_file}"
)

print(
    "\nWARNING: This is synthetic demonstration data."
)
