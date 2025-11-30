"""
Train LSTM with top predictive sensors only
Fixed for Keras 2.13 compatibility
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 70)
print("LSTM WITH TOP PREDICTIVE SENSORS ONLY")
print("=" * 70)

# Top predictive sensors identified from correlation analysis
TOP_SENSORS = [
    'sensor_12', 'sensor_7', 'sensor_21', 'sensor_20', 'sensor_14',
    'sensor_9', 'sensor_13', 'sensor_8', 'sensor_3', 'sensor_17',
    'op_setting_1', 'op_setting_2', 'op_setting_3'
]

print("\n" + "=" * 70)
print("1. LOADING DATA WITH FEATURE SELECTION")
print("=" * 70)

# Load the engineered data
df = pd.read_csv('data/train_FD001_engineered.csv')

# Select only top sensors
feature_cols = [col for col in TOP_SENSORS if col in df.columns]
print(f"✅ Using {len(feature_cols)} features (top sensors only)")
print(f"Features: {', '.join(feature_cols)}")

# Prepare feature matrix
X = df[feature_cols].values
y = df['RUL'].values
unit_ids = df['unit_id'].values

print("\n" + "=" * 70)
print("2. CREATING SEQUENCES")
print("=" * 70)

# Sequence parameters
sequence_length = 30

def create_sequences(X, y, unit_ids, seq_length):
    X_seq, y_seq = [], []
    
    for unit_id in np.unique(unit_ids):
        engine_mask = unit_ids == unit_id
        engine_X = X[engine_mask]
        engine_y = y[engine_mask]
        
        for i in range(len(engine_X) - seq_length + 1):
            X_seq.append(engine_X[i:i + seq_length])
            y_seq.append(engine_y[i + seq_length - 1])
    
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, unit_ids, sequence_length)
print(f"✅ Created sequences:")
print(f"   X shape: {X_seq.shape}")
print(f"   y shape: {y_seq.shape}")

print("\n" + "=" * 70)
print("3. STANDARDIZING FEATURES")
print("=" * 70)

# Reshape for scaling
n_samples, n_timesteps, n_features = X_seq.shape
X_reshaped = X_seq.reshape(-1, n_features)

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# Reshape back
X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
print("✅ Features standardized (mean=0, std=1)")

# Save scaler
os.makedirs('models/trained', exist_ok=True)
scaler_path = 'models/scaler_top_sensors.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Saved scaler: {scaler_path}")

print("\n" + "=" * 70)
print("4. TRAIN/VAL SPLIT")
print("=" * 70)

# Split data
split_idx = int(0.8 * len(X_scaled))
X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

print("\n" + "=" * 70)
print("5. BUILDING OPTIMIZED LSTM")
print("=" * 70)

model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
    layers.Dropout(0.2),
    layers.LSTM(25, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(model.summary())

print("\n" + "=" * 70)
print("6. TRAINING MODEL")
print("=" * 70)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/trained/best_model_top_sensors.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

print("🚀 Training...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 70)
print("7. SAVING MODEL")
print("=" * 70)

# Save final model in .h5 format
model_path = 'models/trained/best_model_top_sensors.h5'
model.save(model_path, save_format='h5')
print(f"✅ Saved model: {model_path}")

print("\n" + "=" * 70)
print("8. FINAL METRICS")
print("=" * 70)

# Evaluate
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)

print(f"Train - Loss: {train_loss:.2f}, MAE: {train_mae:.2f}")
print(f"Val   - Loss: {val_loss:.2f}, MAE: {val_mae:.2f}")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel saved: {model_path}")
print(f"Scaler saved: {scaler_path}")
print("\nReady for deployment! 🚀")
