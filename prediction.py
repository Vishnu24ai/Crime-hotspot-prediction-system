import pandas as pd
import joblib
import folium
import numpy as np
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score, precision_score, f1_score
from folium.plugins import MarkerCluster

# Load crime dataset
def load_data():
    """
    Loads and preprocesses the crime dataset.
    """
    file_path = r"C:\Users\USER\OneDrive\Desktop\CRIMEANALYISI\data\hotspot_data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["latitude", "longitude"])  # Validate missing values
    df["date_time"] = pd.to_datetime(df["date_time"])
    print(f"✅ Loaded dataset with {len(df)} records.")
    return df

# Train DBSCAN model for hotspot detection
def train_dbscan_model(df, eps=0.01, min_samples=5):
    """
    Trains the DBSCAN model and assigns cluster labels.
    """
    print("⚙️ Training DBSCAN model...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["dbscan_cluster"] = dbscan.fit_predict(df[["latitude", "longitude"]])

    model_path = r"C:\Users\USER\Desktop\CRIMEANALYISI\models\dbscan_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(dbscan, model_path)
    print(f"✅ DBSCAN model trained and saved at {model_path}")
    
    return dbscan, df

# Train K-Means model for hotspot detection
def train_kmeans_model(df, n_clusters=5):
    """
    Trains the K-Means model and assigns cluster labels.
    """
    print("⚙️ Training K-Means model...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["kmeans_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])

    model_path = r"C:\Users\USER\Desktop\CRIMEANALYISI\models\kmeans_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(kmeans, model_path)
    print(f"✅ K-Means model trained and saved at {model_path}")

   
    
    return kmeans, df


# Predict hotspot cluster for a new location
def predict_hotspot(lat, lon, model_type="dbscan"):
    """
    Predicts the cluster label for a given location using the chosen model.
    """
    model_paths = {
        "dbscan": r"C:\Users\USER\Desktop\CRIMEANALYISI\models\dbscan_model.pkl",
        "kmeans": r"C:\Users\USER\Desktop\CRIMEANALYISI\models\kmeans_model.pkl"
    }
    
    if model_type not in model_paths:
        print(f"❌ Invalid model type: {model_type}. Choose 'dbscan' or 'kmeans'.")
        return None
    
    model_path = model_paths[model_type]
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please train the model first.")
        return None

    model = joblib.load(model_path)
    
    try:
        if model_type == "dbscan":
            cluster = model.fit_predict(np.array([[lat, lon]]))[0]
        else:  # K-Means
            cluster = model.predict(np.array([[lat, lon]]))[0]
        
        return cluster
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Generate a map with crime hotspots
def generate_map(df):
    """
    Generates and saves a crime hotspot map.
    """
    print("⚙️ Generating crime hotspot map...")
    crime_map = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(crime_map)

    # Add markers to the map
    for _, row in df.iterrows():
        popup_text = f"""
        <b>Crime Type:</b> {row.get('crime_type', 'Unknown')}<br>
        <b>Date:</b> {row['date_time']}
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_text,
            icon=folium.Icon(color="red" if row["dbscan_cluster"] != -1 else "blue")
        ).add_to(marker_cluster)

    # Save the map
    output_path = r"C:\Users\USER\Desktop\CRIMEANALYISI\hotspot_map.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    crime_map.save(output_path)
    print(f"✅ Crime hotspot map saved at {output_path}")

# Main application
if __name__ == "__main__":
    # Load data
    df = load_data()

    # Train models
    dbscan, df = train_dbscan_model(df)
    kmeans, df = train_kmeans_model(df)

    # Generate the map
    generate_map(df)

    # Example predictions
    new_lat, new_lon = 12.9716, 77.5946  # Sample location
    
    dbscan_cluster = predict_hotspot(new_lat, new_lon, "dbscan")
    kmeans_cluster = predict_hotspot(new_lat, new_lon, "kmeans")
    
    if dbscan_cluster is not None:
        print(f"📍 DBSCAN predicted cluster for ({new_lat}, {new_lon}): {dbscan_cluster}")
    
    if kmeans_cluster is not None:
        print(f"📍 K-Means predicted cluster for ({new_lat}, {new_lon}): {kmeans_cluster}")
