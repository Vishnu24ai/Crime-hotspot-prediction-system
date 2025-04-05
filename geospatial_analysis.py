import pandas as pd
import folium
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Load dataset
def load_data():
    file_path = r"C:\Users\USER\Desktop\CRIMEANALYISI\data\crime_data (1).csv"
    df = pd.read_csv(file_path)
    required_columns = ["latitude", "longitude"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"✅ Loaded dataset with {len(df)} records.")
    return df

# Dynamic eps calculation
def calculate_dynamic_eps(df, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(df[['latitude', 'longitude']])
    distances, _ = nbrs.kneighbors(df[['latitude', 'longitude']])
    return np.mean(distances[:, -1])  # Average distance to the k-th nearest neighbor

# DBSCAN clustering
def dbscan_clustering(df, eps, min_samples):
    print(f"⚙️ Running DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['dbscan_cluster'] = dbscan.fit_predict(df[['latitude', 'longitude']])
    print(f"✅ Clustering complete. Noise points: {sum(df['dbscan_cluster'] == -1)}")
    return df

# Generate crime hotspot map
def generate_map(df):
    crime_map = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)

    # Colors for clusters
    colors = ["red", "blue", "green", "purple", "orange", "yellow", "pink", "cyan", "brown"]
    dbscan_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(set(df["dbscan_cluster"]))}

    # Add markers to the map
    max_markers = 1000
    if len(df) > max_markers:
        df = df.sample(n=max_markers, random_state=42)  # Limit markers for performance

    for _, row in df.iterrows():
        color = dbscan_colors[row["dbscan_cluster"]] if row["dbscan_cluster"] != -1 else "black"  # Noise points in black
        tooltip = "Noise Point" if row["dbscan_cluster"] == -1 else f"Cluster {row['dbscan_cluster']}"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            tooltip=tooltip
        ).add_to(crime_map)

    # Dynamic legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                z-index:9999; font-size:14px; 
                background:white; padding:10px; 
                border:1px solid black;">
        <b>DBSCAN Crime Hotspot Map Legend</b><br>
    """
    for cluster, color in dbscan_colors.items():
        cluster_label = "Noise" if cluster == -1 else f"Cluster {cluster}"
        legend_html += f'<i style="color:{color}">●</i>: {cluster_label}<br>'
    legend_html += "</div>"
    crime_map.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    map_path = r"C:\Users\USER\Desktop\CRIMEANALYISI\templates\crime_map.html"
    crime_map.save(map_path)
    print(f"✅ DBSCAN Crime Hotspot Map generated! Check {map_path}")

if __name__ == "__main__":
    df = load_data()
    eps = calculate_dynamic_eps(df)  # Dynamically calculate eps
    df = dbscan_clustering(df, eps=eps, min_samples=10)
    df.to_csv(r"C:\Users\USER\Desktop\CRIMEANALYISI\data\dbscan_hotspots.csv", index=False)
    generate_map(df)
