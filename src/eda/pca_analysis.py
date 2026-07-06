import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# PCA ANALYSIS
# ==========================================================

def pca_analysis(
        df,
        sample_size=10000,
        random_state=42):

    print("\n" + "="*70)
    print("PCA ANALYSIS")
    print("="*70)

    create_directories()

    if "Label" not in df.columns:
        raise Exception("Label column not found.")

    # ------------------------------------------------------
    # Sampling
    # ------------------------------------------------------

    if len(df) > sample_size:

        sampled_df = df.sample(
            n=sample_size,
            random_state=random_state
        )

    else:

        sampled_df = df.copy()

    # ------------------------------------------------------
    # Features / Labels
    # ------------------------------------------------------

    X = sampled_df.drop("Label", axis=1)

    y = sampled_df["Label"]

    print(f"\nSamples Used : {len(X):,}")
    print(f"Features     : {X.shape[1]}")

    # ------------------------------------------------------
    # Standardization
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    pca = PCA()

    X_pca = pca.fit_transform(X_scaled)

    # ------------------------------------------------------
    # Save PCA Model
    # ------------------------------------------------------

    joblib.dump(

        pca,

        os.path.join(
            TABLE_PATH,
            "pca_model.pkl"
        )

    )

    # ------------------------------------------------------
    # Explained Variance
    # ------------------------------------------------------

    variance = pca.explained_variance_ratio_

    variance_df = pd.DataFrame({

        "Principal Component":

            np.arange(
                1,
                len(variance)+1
            ),

        "Explained Variance":

            variance,

        "Cumulative Variance":

            np.cumsum(variance)

    })

    variance_df.to_csv(

        os.path.join(
            TABLE_PATH,
            "explained_variance.csv"
        ),

        index=False

    )

    # ------------------------------------------------------
    # 2D PCA
    # ------------------------------------------------------

    plt.figure(figsize=(10,7))

    scatter = plt.scatter(

        X_pca[:,0],

        X_pca[:,1],

        c=y,

        s=8,

        alpha=0.7

    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("PCA (2D Projection)")

    plt.colorbar(scatter)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "pca_2d.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # 3D PCA
    # ------------------------------------------------------

    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    scatter = ax.scatter(

        X_pca[:,0],

        X_pca[:,1],

        X_pca[:,2],

        c=y,

        s=6,

        alpha=0.6

    )

    ax.set_xlabel("PC1")

    ax.set_ylabel("PC2")

    ax.set_zlabel("PC3")

    plt.title("3D PCA")

    plt.colorbar(scatter)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "pca_3d.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Scree Plot
    # ------------------------------------------------------

    plt.figure(figsize=(10,6))

    plt.plot(

        np.arange(
            1,
            len(variance)+1
        ),

        variance,

        marker="o"

    )

    plt.xlabel("Principal Component")

    plt.ylabel("Explained Variance")

    plt.title("Scree Plot")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "scree_plot.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Cumulative Variance
    # ------------------------------------------------------

    plt.figure(figsize=(10,6))

    plt.plot(

        np.arange(
            1,
            len(variance)+1
        ),

        np.cumsum(variance),

        marker="o"

    )

    plt.xlabel("Principal Component")

    plt.ylabel("Cumulative Explained Variance")

    plt.grid(True)

    plt.title("Cumulative Explained Variance")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_PATH,

            "cumulative_variance.png"

        ),

        dpi=300

    )

    plt.close()

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    summary = {

        "Total Samples":

            int(len(X)),

        "Original Features":

            int(X.shape[1]),

        "Principal Components":

            int(len(variance)),

        "Variance (PC1)":

            float(variance[0]),

        "Variance (PC2)":

            float(variance[1]),

        "Variance (PC3)":

            float(variance[2]),

        "Variance First 10":

            float(
                np.cumsum(variance)[9]
            )

    }

    with open(

        os.path.join(

            TABLE_PATH,

            "pca_summary.json"

        ),

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4

        )

    # ------------------------------------------------------
    # Console
    # ------------------------------------------------------

    print("\nExplained Variance")

    print(variance_df.head(10))

    print("\nGenerated Files")

    print("------------------------------------------")

    print("explained_variance.csv")

    print("pca_summary.json")

    print("pca_model.pkl")

    print("pca_2d.png")

    print("pca_3d.png")

    print("scree_plot.png")

    print("cumulative_variance.png")

    print("------------------------------------------")

    print("\nPCA Analysis Completed Successfully.")

    return variance_df


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    from loader import load_dataset
    from config import PROCESSED_DATASET

    df = load_dataset(PROCESSED_DATASET)

    pca_analysis(df)
    