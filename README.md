# Credit Card Customer Segmentation (K-Means)

This project performs **customer segmentation** on a credit card dataset using **K-Means clustering**.

## 🚀 What it does
- Loads the `CC GENERAL.csv` dataset (8,950 customers, 18 features)
- Dataset source: **Kaggle (Credit Card Customer Data)**
- Performs **EDA** (distribution, missing values, correlations, outliers)
- Imputes missing values, scales features with **StandardScaler**
- Finds the optimal number of clusters using **Elbow + Silhouette**
- Trains a **K-Means** model and assigns cluster labels
- Profiles cluster behavior with means, heatmaps, and radar charts
- Exports a fully segmented CSV (`CC_Segmented.csv`)

## ✅ Key results
- The final model uses **K = 3** clusters.
- The project produces:
  - `CC_Segmented.csv` (customer + cluster + segment label)
  - Notebook visualization outputs (histograms, correlation heatmap, cluster profiles, PCA plot)

## ▶️ How to run
1. Install dependencies (recommended in a venv):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook:
   - Open `Kmeans/model.ipynb` in VS Code / Jupyter
   - Run cells sequentially (or `Run All`)

## 📌 Notes
- Update the cluster names in the notebook (`CLUSTER_NAMES`) after reviewing the profile tables/heatmaps.
- The exported CSV is saved as `CC_Segmented.csv`.

## 📂 Files
- `Kmeans/model.ipynb` – main notebook
- `CC GENERAL.csv` – original data (not committed if using .gitignore)
- `CC_Segmented.csv` – output dataset (regenerated).

---

