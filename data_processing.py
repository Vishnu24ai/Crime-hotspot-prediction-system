import pandas as pd

def load_and_clean_data(file_path):
    """Load crime dataset and preprocess it."""
    df = pd.read_csv(r"C:\Users\USER\Desktop\CRIMEANALYISI\data\crime_data (1).csv")

    # Drop duplicates
    df = df.drop_duplicates()

    # Convert date_time column to datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Extract new features: day_of_week, hour
    df['day_of_week'] = df['date_time'].dt.day_name()
    df['hour'] = df['date_time'].dt.hour

    # Handle missing values 
    df['weather'].fillna(df['weather'].mode()[0], inplace=True)
    df['police_stations_nearby'].fillna(df['police_stations_nearby'].median(), inplace=True)

    # Save the cleaned dataset
    df.to_csv(r"C:\Users\USER\Desktop\CRIMEANALYISI\data\cleaned_crime_data.csv", index=False)

    print("✅ Data preprocessing complete. Cleaned dataset saved!")

    return df

if __name__ == "__main__":
    file_path = "../data/crime_data.csv"
    cleaned_data = load_and_clean_data(file_path)
