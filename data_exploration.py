import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\data\crime_data (1).csv")

# Convert 'date_time' to datetime
df["date_time"] = pd.to_datetime(df["date_time"])

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# 📌 1. Crime Type Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="crime_type", order=df["crime_type"].value_counts().index, palette="viridis")
plt.xticks(rotation=45)
plt.title("Crime Type Distribution")
plt.savefig(r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\static\crime_type_distribution.png")
plt.show()


# Convert to year-month period for time aggregation
df["year_month"] = df["date_time"].dt.to_period("M")

# Plot crime trends over time
plt.figure(figsize=(12, 6))  
crime_trends = df["year_month"].value_counts().sort_index()

# Plot with improved styling
crime_trends.plot(kind="line", marker="o", color="red", linewidth=2, alpha=0.8)
plt.grid(True, linestyle="--", alpha=0.5)  # Add gridlines for better readability
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.title("Crime Trends Over Time", fontsize=16)
plt.xlabel("Year-Month", fontsize=12)
plt.ylabel("Number of Crimes", fontsize=12)
plt.tight_layout()  # Adjust layout to avoid clipping
plt.savefig(r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\static\crime_trends.png")  # Save the plot
plt.show()  # Display the plot

# 📌 3. K-Means Clustering for Crime Hotspots
k = 5  # Number of clusters 
df = df.dropna(subset=["latitude", "longitude"])  # Remove missing coordinates

X = df[["latitude", "longitude"]].values
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# 📌 4. Visualizing Clusters on a Folium Map
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)

colors = ["red", "blue", "green", "purple", "orange"]  # Different colors for clusters
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=colors[row["Cluster"]],
        fill=True,
        fill_color=colors[row["Cluster"]],
        fill_opacity=0.6,
    ).add_to(m)

# Save the crime cluster map
map_path = r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\templates\crime_clusters.html"
m.save(map_path)
print(f"✅ Crime Clustering Map Generated! Check {map_path}")
