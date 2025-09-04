import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load dataset
df = pd.read_csv("Indian_Kids_Screen_Time.csv")

# 2. Feature Engineering
df["screen_time_per_year_of_age"] = df["Avg_Daily_Screen_Time_hr"] / df["Age"]

# Define feature sets
num_features = ["Age", "Avg_Daily_Screen_Time_hr", 
                "Educational_to_Recreational_Ratio", 
                "screen_time_per_year_of_age"]
cat_features = ["Gender", "Primary_Device", "Health_Impacts", "Urban_or_Rural"]

# 3. Preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

# 4. Separate PCA pipeline (for visualization)
pca_only = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=2))
])

pca_result = pca_only.fit_transform(df)
pca_data = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

# 5. Full pipeline with clustering
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=2)),
    ("cluster", KMeans(n_clusters=3, random_state=42, n_init=10))
])

pipeline.fit(df)
clusters = pipeline.named_steps["cluster"].labels_
pca_data["Cluster"] = clusters

# 6. Evaluation (Silhouette Score)
silhouette = silhouette_score(pca_result, clusters)
print(f"Silhouette Score (cluster quality): {silhouette:.3f}")

# 7. Visualization
os.makedirs("outputs", exist_ok=True)

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[num_features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# PCA scatter plot with clusters
plt.figure(figsize=(6,5))
sns.scatterplot(data=pca_data, x="PC1", y="PC2", hue="Cluster", palette="Set2")
plt.title("PCA Scatter Plot with KMeans Clusters")
plt.savefig("outputs/pca_clusters.png")
plt.show()

#silhouette score
with open("outputs/metrics.txt", "w") as f:
    f.write(f"Silhouette Score (cluster quality): {silhouette:.3f}\n")
