# Scaler Customer Segmentation 🛍️

This repository contains a business case study for **Scaler**, focusing on **customer segmentation** using unsupervised machine learning. The project groups customers into distinct clusters based on their purchasing behavior and demographics.

By identifying these customer archetypes, this analysis helps Scaler move from a one-size-fits-all approach to **targeted marketing**, allowing for personalized communication, optimized promotions, and improved customer retention.

---

## 🎯 Project Goal

The primary objective is to apply clustering algorithms to the Scaler dataset to discover and analyze different customer segments. The key business question is: **"What distinct groups of customers does Scaler have, and how do their behaviors differ?"**

## 🛠️ Tech Stack

This project utilizes a complete stack for unsupervised learning and data analysis:

* **Data Analysis:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (specifically for PCA, KMeans, Hierarchical Clustering, DBSCAN, and scaling)

---

## 🚀 Project Workflow

This analysis follows a structured unsupervised learning pipeline:

### 1. Data Cleaning & Exploration (EDA)
* Loaded the `scaler_clustering.csv` dataset.
* Checked for and handled any missing values or duplicate entries.
* Performed deep **Univariate and Bivariate Analysis** to understand the distributions and relationships of features like `orders_placed`, `total_spend`, `age`, etc.
* A correlation heatmap was generated to check for multicollinearity.

### 2. Data Preprocessing
* This was a critical step for clustering:
    * **Outlier Treatment:** Identified and handled outliers to prevent them from skewing the cluster centers.
    * **Feature Scaling:** Applied **StandardScaler** and **MinMaxScaler** to normalize the data, ensuring all features contribute equally to the distance calculations.

### 3. Dimensionality Reduction (PCA)
* **Principal Component Analysis (PCA)** was applied to the scaled data.
* This step reduces the number of features, combats the "curse of dimensionality," and makes it possible to visualize the clusters in 2D or 3D.

### 4. Model Building & Cluster Identification
* Several clustering algorithms were built and compared to find the most logical and stable groupings:
    * **KMeans Clustering:**
        * The **Elbow Method** was used to find the optimal number of clusters (k).
        * The **Silhouette Score** was calculated to evaluate the density and separation of the resulting clusters.
    * **Agglomerative (Hierarchical) Clustering:**
        * A **dendrogram** was plotted to visualize the hierarchical relationships and help determine the number of clusters.
    * **DBSCAN:**
        * This density-based algorithm was used to identify clusters of varying shapes and sizes and to effectively isolate noise/outliers.

### 5. Cluster Profiling & Analysis
* After finalizing the clusters with KMeans, the different segments were analyzed:
* Calculated the mean/median of each feature for each cluster (e.g., "Cluster 0: High Spend, Low Frequency," "Cluster 1: Low Spend, High Frequency").
* Visualized the clusters using PCA plots to clearly see their separation.
* This final step provides the actionable **business insights** on who each customer segment is and how Scaler should engage with them.

---

## 📂 How to Use

To run this analysis yourself:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Scaler Clustering Business Case.ipynb"
    ```
