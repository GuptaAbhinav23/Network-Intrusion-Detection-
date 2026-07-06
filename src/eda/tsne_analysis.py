"""
tsne_analysis.py

t-SNE Analysis for Network Intrusion Detection Dataset

Author : Abhinav Gupta
Project : Network Intrusion Detection System
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import TABLE_PATH, FIGURE_PATH


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

def create_directories():

    os.makedirs(TABLE_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)


# ==========================================================
# t-SNE ANALYSIS
# ==========================================================

def tsne_analysis(
        df: pd.DataFrame,
        sample_size=10000,
        pca_components=50,
        random_state=42):

    print("\n" + "=" * 70)
    print("t-SNE ANALYSIS")
    print("=" * 70)

    create_directories()

    if "Label" not in df.columns:
        raise Exception("Label column not found.")

    # ------------------------------------------------------
    # Stratified Sampling
    # ------------------------------------------------------

    if len(df) > sample_size:
        samples = []

        for _, group in df.groupby("Label"):
            n = max(1, int(sample_size * len(group) / len(df)))
            samples.append(group.sample(n=n, random_state=random_state))

        sampled_df = (
            pd.concat(samples)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    else:

        sampled_df = df.copy()

    print(f"\nSamples Used : {len(sampled_df):,}")

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    X = sampled_df.drop("Label", axis=1)

    y = sampled_df["Label"]

    # ------------------------------------------------------
    # Standard Scaling
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------
    # PCA before t-SNE
    # ------------------------------------------------------

    pca = PCA(
        n_components=min(
            pca_components,
            X_scaled.shape[1]
        ),
        random_state=random_state
    )

    X_pca = pca.fit_transform(X_scaled)

    print("PCA completed.")

    # ------------------------------------------------------
    # t-SNE 2D
    # ------------------------------------------------------

    print("Running 2D t-SNE...")

    tsne2 = TSNE(

        n_components=2,

        perplexity=30,

        learning_rate="auto",

        init="pca",

        random_state=random_state

    )

    embedding2 = tsne2.fit_transform(X_pca)

    embedding2_df = pd.DataFrame({

        "TSNE1": embedding2[:, 0],

        "TSNE2": embedding2[:, 1],

        "Label": y.values

    })

    embedding2_df.to_csv(

        os.path.join(

            TABLE_PATH,

            "tsne_2d_embedding.csv"

        ),

        index=False

    )

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(

        embedding2[:, 0],

        embedding2[:, 1],

        c=y,

        s=8,

        alpha=0.7

    )

    plt.xlabel("t-SNE 1")

    plt.ylabel("t-SNE 2")

    plt.title("2D t-SNE Projection")

    plt.colorbar(scatter)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "tsne_2d.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # t-SNE 3D
    # ------------------------------------------------------

    print("Running 3D t-SNE...")

    tsne3 = TSNE(

        n_components=3,

        perplexity=30,

        learning_rate="auto",

        init="pca",

        random_state=random_state

    )

    embedding3 = tsne3.fit_transform(X_pca)

    embedding3_df = pd.DataFrame({

        "TSNE1": embedding3[:, 0],

        "TSNE2": embedding3[:, 1],

        "TSNE3": embedding3[:, 2],

        "Label": y.values

    })

    embedding3_df.to_csv(

        os.path.join(

            TABLE_PATH,

            "tsne_3d_embedding.csv"

        ),

        index=False

    )

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    scatter = ax.scatter(

        embedding3[:, 0],

        embedding3[:, 1],

        embedding3[:, 2],

        c=y,

        s=6,

        alpha=0.6

    )

    ax.set_xlabel("TSNE1")

    ax.set_ylabel("TSNE2")

    ax.set_zlabel("TSNE3")

    plt.title("3D t-SNE Projection")

    plt.colorbar(scatter)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "tsne_3d.png"

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
        "PCA Components": int(X_pca.shape[1]),
        "Perplexity": 30

    }

    with open(

        os.path.join(

            TABLE_PATH,

            "tsne_summary.json"

        ),

        "w"

    ) as f:

        json.dump(

            summary,

            f,

            indent=4

        )

    print("\nGenerated Files")

    print("-------------------------------------------")

    print("tsne_2d_embedding.csv")
    print("tsne_3d_embedding.csv")
    print("tsne_summary.json")
    print("tsne_2d.png")
    print("tsne_3d.png")

    print("-------------------------------------------")

    print("\nt-SNE Analysis Completed Successfully.")

    return embedding2_df