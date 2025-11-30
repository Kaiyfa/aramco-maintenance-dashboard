"""
Recreate scaler from training data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 70)
print("RECREATING SCALER FROM TRAINING DATA")
print("=" * 70)

# Top sensors used in model
TOP_SENSORS = [
    'sensor_12', 'sensor_7', 'sensor_21', 'sensor_20', 'sensor_14',
    'sensor_9', 'sensor_13', 'sensor_8', 'sensor_3', 'sensor_17',
    'op_setting_1', 'op_setting_2', 'op_setting_3'
]

# Load training data
print("\n📂 Loading training data...")
df = pd.read_csv('data/train_FD001_engineered.csv')

# Select features
feature_cols = [col for col in TOP_SENSORS if col in df.columns]
print(f"✅ Using {len(feature_cols)} features")

X = df[feature_cols].values

# Create and fit scaler
print("\n🔧 Creating scaler...")
scaler = StandardScaler()
scaler.fit(X)

print(f"✅ Scaler fitted")
print(f"   Features: {scaler.n_features_in_}")
print(f"   Mean shape: {scaler.mean_.shape}")
print(f"   Scale shape: {scaler.scale_.shape}")

# Save scaler
output_path = 'ml_models/scaler_top_sensors.pkl'
print(f"\n💾 Saving scaler to: {output_path}")
joblib.dump(scaler, output_path)

# Test reload
print("\n🧪 Testing reload...")
test_scaler = joblib.load(output_path)
print(f"✅ Scaler reloads correctly!")
print(f"   Features: {test_scaler.n_features_in_}")

print("\n" + "=" * 70)
print("✅ SCALER RECREATED SUCCESSFULLY!")
print("=" * 70)

