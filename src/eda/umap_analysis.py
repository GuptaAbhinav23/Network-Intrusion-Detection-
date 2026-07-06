"""
umap_analysis.py

UMAP Analysis for Network Intrusion Detection Dataset

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import umap

from .config import TABLE_PATH, FIGURE_PATH


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

def create_directories():

    os.makedirs(TABLE_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)


# ==========================================================
# UMAP ANALYSIS
# ==========================================================

def umap_analysis(
        df,
        sample_size=10000,
        n_neighbors=15,
        min_dist=0.10,
        random_state=42):

    print("\n" + "=" * 70)
    print("UMAP ANALYSIS")
    print("=" * 70)

    create_directories()

    if "Label" not in df.columns:
        raise Exception("Label column not found.")

    # ------------------------------------------------------
    # Stratified Sampling
    # ------------------------------------------------------

    if len(df) > sample_size:

        sampled_df = (
            df.groupby("Label", group_keys=False)
              .apply(
                    lambda x: x.sample(
                        max(
                            1,
                            int(sample_size * len(x) / len(df))
                        ),
                        random_state=random_state
                    )
              )
              .reset_index(drop=True)
        )

    else:

        sampled_df = df.copy()

    print(f"\nSamples Used : {len(sampled_df):,}")

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    X = sampled_df.drop(columns=["Label"])

    y = sampled_df["Label"]

    # ------------------------------------------------------
    # Standard Scaling
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------
    # UMAP
    # ------------------------------------------------------

    print("Running UMAP...")

    reducer = umap.UMAP(

        n_components=2,

        n_neighbors=n_neighbors,

        min_dist=min_dist,

        metric="euclidean",

        random_state=random_state

    )

    embedding = reducer.fit_transform(X_scaled)

    # ------------------------------------------------------
    # Save Embedding
    # ------------------------------------------------------

    embedding_df = pd.DataFrame({

        "UMAP1": embedding[:, 0],

        "UMAP2": embedding[:, 1],

        "Label": y.values

    })

    embedding_df.to_csv(

        os.path.join(

            TABLE_PATH,

            "umap_embedding.csv"

        ),

        index=False

    )

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    joblib.dump(

        reducer,

        os.path.join(

            TABLE_PATH,

            "umap_model.pkl"

        )

    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plt.figure(figsize=(10,8))

    scatter = plt.scatter(

        embedding[:,0],

        embedding[:,1],

        c=y,

        s=8,

        alpha=0.7

    )

    plt.xlabel("UMAP Component 1")

    plt.ylabel("UMAP Component 2")

    plt.title("2D UMAP Projection")

    plt.colorbar(scatter)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "umap_projection.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = {

        "Samples Used": int(len(sampled_df)),
        "Original Features": int(X.shape[1]),
        "Components": 2,
        "Neighbors": int(n_neighbors),
        "Minimum Distance": float(min_dist)

    }

    with open(

        os.path.join(

            TABLE_PATH,

            "umap_summary.json"

        ),

        "w"

    ) as f:

        json.dump(

            summary,

            f,

            indent=4

        )

    print("\nGenerated Files")

    print("----------------------------------------")

    print("umap_embedding.csv")
    print("umap_model.pkl")
    print("umap_summary.json")
    print("umap_projection.png")

    print("----------------------------------------")

    print("\nUMAP Analysis Completed Successfully.")

    return embedding_df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from loader import load_dataset
    from config import PROCESSED_DATASET

    df = load_dataset(PROCESSED_DATASET)

    umap_analysis(df)