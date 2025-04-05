import pandas as pd
import folium
import numpy as np
from folium.plugins import MarkerCluster, HeatMap
import os

# Haversine formula for distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    a = np.sin(d_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Load dataset
file_path = r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\data\hotspot_data.csv"
df = pd.read_csv(file_path)

# Validation for missing latitude or longitude
df = df.dropna(subset=["latitude", "longitude"])

# Round coordinates for precision consistency
df["latitude"] = df["latitude"].round(5)
df["longitude"] = df["longitude"].round(5)

# Normalize severity for heat intensity (if available)
if "severity" in df.columns:
    df["severity"] = df["severity"].fillna(1)  # Default severity = 1 if missing
    max_severity = df["severity"].max()
    df["normalized_severity"] = df["severity"] / max_severity if max_severity > 0 else 1
else:
    df["normalized_severity"] = 1  # Default intensity if severity is missing

# Dynamic map centering and bounds fitting
crime_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=12)
bounds = [[df.latitude.min(), df.longitude.min()], [df.latitude.max(), df.longitude.max()]]
crime_map.fit_bounds(bounds)

# Add clustered crime locations to the map
marker_cluster = MarkerCluster().add_to(crime_map)
for _, row in df.iterrows():
    popup_text = f"""
    <b>Crime Type:</b> {row.get('crime_type', 'N/A')}<br>
    <b>Description:</b> {row.get('description', 'No details available')}<br>
    <b>Date:</b> {row.get('date_time', 'Unknown')}<br>
    <b>Severity:</b> {row.get('severity', 'Unknown')}<br>
    <b>Status:</b> {row.get('status', 'Pending')}
    """
    folium.Marker(
        location=[row.latitude, row.longitude],
        popup=popup_text,
        icon=folium.Icon(color="red" if row.get('prediction_label', 0) == 1 else "blue")
    ).add_to(marker_cluster)

# Add heatmap for crime density (adjusted for severity)
heat_data = [[row["latitude"], row["longitude"], row["normalized_severity"]] for _, row in df.iterrows()]
HeatMap(heat_data, min_opacity=0.4, radius=10, blur=15, max_zoom=13).add_to(crime_map)

# Add legend for marker colors
legend_html = """
<div style="
position: fixed; 
bottom: 20px; 
left: 20px; 
z-index: 1000; 
background-color: white; 
padding: 10px; 
border: 2px solid black; 
border-radius: 5px;
box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
">
<b>Crime Marker Legend</b><br>
<i style="color:red">Red</i>: Predicted Hotspot<br>
<i style="color:blue">Blue</i>: Non-Hotspot<br>
<i style="background: linear-gradient(to right, rgba(255, 0, 0, 0.8), rgba(255, 255, 0, 0.8))">
Heatmap Intensity (Low → High)</i>
</div>
"""
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save map as HTML
output_path = r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\hotspot_map.html"
try:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    crime_map.save(output_path)
    print("✅ Crime Hotspot Map saved as 'hotspot_map.html'")
except Exception as e:
    print(f"Error saving map: {e}")
