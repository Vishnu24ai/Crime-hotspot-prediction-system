import pandas as pd
import folium
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import folium_static
from sklearn.cluster import DBSCAN, KMeans
from folium.plugins import MarkerCluster, HeatMap

# Load crime dataset
def load_data():
    file_path = r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\data\hotspot_data.csv"
    try:
        df = pd.read_csv(file_path)
        df["date_time"] = pd.to_datetime(df["date_time"])
        df = df.dropna(subset=["latitude", "longitude"])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    return df

# Train clustering models
def train_hotspot_models(df, eps=0.01, min_samples=5, n_clusters=5):
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df["dbscan_cluster"] = dbscan.fit_predict(df[["latitude", "longitude"]])
        joblib.dump(dbscan, r"C:\Users\USER\Desktop\CRIMEANALYISI\models\dbscan_model.pkl")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["kmeans_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])
        joblib.dump(kmeans, r"C:\Users\USER\Desktop\CRIMEANALYISI\models\kmeans_model.pkl")
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, df
    return dbscan, kmeans, df

# Generate default map
def generate_default_map(df, enable_heatmap=False):
    crime_map = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)

    if enable_heatmap:
        heat_data = [[row["latitude"], row["longitude"]] for _, row in df.iterrows()]
        HeatMap(heat_data).add_to(crime_map)
    else:
        marker_cluster = MarkerCluster().add_to(crime_map)
        for _, row in df.iterrows():
            popup_text = f"""
            <b>Coordinates:</b> ({row['latitude']}, {row['longitude']})<br>
            <b>Crime Type:</b> {row.get('crime_type', 'Unknown')}<br>
            <b>Date:</b> {row['date_time']}<br>
            <b>Description:</b> {row.get('description', 'No details available')}
            """
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=popup_text,
                icon=folium.Icon(color="red" if row["dbscan_cluster"] != -1 else "blue")
            ).add_to(marker_cluster)

    return crime_map

# ✅ Updated function with coordinate tooltip
def generate_hotspot_map(df, selected_lat, selected_lon, radius=1000):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        d_phi = np.radians(lat2 - lat1)
        d_lambda = np.radians(lon2 - lon1)
        a = np.sin(d_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    df["distance"] = df.apply(lambda row: haversine(selected_lat, selected_lon, row["latitude"], row["longitude"]), axis=1)
    nearby_crimes = df[df["distance"] <= radius]

    hotspot_map = folium.Map(location=[selected_lat, selected_lon], zoom_start=12)

    for _, row in nearby_crimes.iterrows():
        popup_text = f"""
        <b>Coordinates:</b> ({row['latitude']}, {row['longitude']})<br>
        <b>Crime Type:</b> {row.get('crime_type', 'Unknown')}<br>
        <b>Date:</b> {row['date_time']}<br>
        <b>Description:</b> {row.get('description', 'No details available')}
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="red" if row["dbscan_cluster"] != -1 else "blue",
            fill=True,
            fill_color="red" if row["dbscan_cluster"] != -1 else "blue",
            fill_opacity=0.7,
            popup=popup_text,
            tooltip=f"({row['latitude']:.5f}, {row['longitude']:.5f})"  # ← Added tooltip here
        ).add_to(hotspot_map)

    return hotspot_map, nearby_crimes

# Display crime visualizations
def display_visualizations(nearby_crimes):
    if nearby_crimes.empty:
        st.warning("No crime data available for the selected hotspot area.")
        return

    st.subheader("Crime Type Breakdown in Selected Hotspot Area")
    crime_types = nearby_crimes["crime_type"].value_counts()
    st.bar_chart(crime_types)

    st.subheader("Crime Trend Over Time in Selected Hotspot Area")
    nearby_crimes["date"] = nearby_crimes["date_time"].dt.date
    crimes_by_date = nearby_crimes["date"].value_counts().sort_index()
    st.line_chart(crimes_by_date)

    st.subheader("Crime Distribution in Selected Hotspot Area")
    fig, ax = plt.subplots()
    crime_types.plot.pie(autopct='%1.1f%%', startangle=90, cmap="tab10", ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

# Main app
def main():
    st.title("Crime Hotspot Prediction System")
    df = load_data()
    if df is None:
        return

    dbscan, kmeans, df = train_hotspot_models(df)
    if dbscan is None or kmeans is None:
        return

    enable_heatmap = st.checkbox("Enable Heatmap", value=False)

    st.subheader("Crime Map")
    default_map = generate_default_map(df, enable_heatmap=enable_heatmap)
    folium_static(default_map)

    st.subheader("Select a Location for Hotspot Analysis")
    selected_lat = st.number_input("Latitude", value=df["latitude"].mean())
    selected_lon = st.number_input("Longitude", value=df["longitude"].mean())

    if st.button("Analyze Hotspot"):
        hotspot_map, nearby_crimes = generate_hotspot_map(df, selected_lat, selected_lon)
        st.subheader("Hotspot Map")
        folium_static(hotspot_map)

        display_visualizations(nearby_crimes)

if __name__ == "__main__":
    main()
