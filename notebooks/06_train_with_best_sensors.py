"""
Train LSTM using only the most predictive sensors
Simpler features = better generalization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("LSTM WITH TOP PREDICTIVE SENSORS ONLY")
print("=" * 70)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. LOAD ORIGINAL DATA AND SELECT TOP SENSORS
# ============================================================
print("\n" + "=" * 70)
print("1. LOADING DATA WITH FEATURE SELECTION")
print("=" * 70)

# Load processed data
df = pd.read_csv('data/processed/train_FD001_with_RUL.csv')

# Top 10 sensors from EDA correlation analysis
top_sensors = [
    'sensor_12', 'sensor_7', 'sensor_21', 'sensor_20',
    'sensor_14', 'sensor_9', 'sensor_13', 'sensor_8',
    'sensor_3', 'sensor_17'
]

# Also include operational settings
feature_cols = top_sensors + ['op_setting_1', 'op_setting_2', 'op_setting_3']

print(f"✅ Using {len(feature_cols)} features (top sensors only)")
print(f"Features: {', '.join(feature_cols)}")

# Apply RUL clipping
df['RUL_clipped'] = df['RUL'].clip(upper=125)

# ============================================================
# 2. CREATE SEQUENCES WITH SELECTED FEATURES
# ============================================================
print("\n" + "=" * 70)
print("2. CREATING SEQUENCES")
print("=" * 70)

SEQUENCE_LENGTH = 30

def create_sequences(data, unit_col, feature_cols, target_col, seq_length):
    sequences = []
    targets = []
    
    for unit_id in data[unit_col].unique():
        unit_data = data[data[unit_col] == unit_id]
        features = unit_data[feature_cols].values
        target = unit_data[target_col].values
        
        for i in range(len(features) - seq_length + 1):
            sequences.append(features[i:i + seq_length])
            targets.append(target[i + seq_length - 1])
    
    return np.array(sequences), np.array(targets)

X, y = create_sequences(df, 'unit_id', feature_cols, 'RUL_clipped', SEQUENCE_LENGTH)

print(f"✅ Created sequences:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# ============================================================
# 3. STANDARDIZE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("3. STANDARDIZING FEATURES")
print("=" * 70)

# Reshape for scaling
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, n_timesteps, n_features)

print("✅ Features standardized (mean=0, std=1)")

# Save scaler
scaler_path = Path('models/scaler_top_sensors.pkl')
import joblib
joblib.dump(scaler, scaler_path)
print(f"✅ Saved scaler: {scaler_path}")

# ============================================================
# 4. TRAIN/VAL SPLIT
# ============================================================
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# ============================================================
# 5. BUILD OPTIMIZED MODEL
# ============================================================
print("\n" + "=" * 70)
print("5. BUILDING OPTIMIZED LSTM")
print("=" * 70)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols)),
         kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    LSTM(25, return_sequences=False, kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    Dense(10, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1)
])

print(model.summary())

# ============================================================
# 6. COMPILE AND TRAIN
# ============================================================
print("\n" + "=" * 70)
print("6. TRAINING MODEL")
print("=" * 70)

model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss='mse',
    metrics=['mae']
)

models_dir = Path('models/trained')
models_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(models_dir / 'best_model_top_sensors.keras'), 
                   monitor='val_loss', save_best_only=True, verbose=1)
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

# ============================================================
# 7. EVALUATE
# ============================================================
print("\n" + "=" * 70)
print("7. EVALUATING PERFORMANCE")
print("=" * 70)

y_train_pred = model.predict(X_train, verbose=0).flatten()
y_val_pred = model.predict(X_val, verbose=0).flatten()

# Clip predictions
y_train_pred = np.clip(y_train_pred, 0, 125)
y_val_pred = np.clip(y_val_pred, 0, 125)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("\n📊 TRAINING PERFORMANCE:")
print(f"   RMSE: {train_rmse:.2f} cycles")
print(f"   MAE:  {train_mae:.2f} cycles")
print(f"   R²:   {train_r2:.4f}")

print("\n📊 VALIDATION PERFORMANCE:")
print(f"   RMSE: {val_rmse:.2f} cycles")
print(f"   MAE:  {val_mae:.2f} cycles")
print(f"   R²:   {val_r2:.4f}")

if val_r2 > 0.6 and val_rmse < 25:
    print("\n🌟 EXCELLENT! Model ready for deployment!")
elif val_r2 > 0.4 and val_rmse < 35:
    print("\n✅ GOOD! Acceptable for prototype!")
else:
    print("\n⚠️  Needs improvement, but usable for demo")

# ============================================================
# 8. VISUALIZE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[0, 0].set_title('Training History', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_val, y_val_pred, alpha=0.4, s=15)
axes[0, 1].plot([0, 125], [0, 125], 'r--', linewidth=2)
axes[0, 1].set_title(f'Predictions (R²={val_r2:.3f}, RMSE={val_rmse:.1f})', fontweight='bold')
axes[0, 1].set_xlabel('Actual RUL')
axes[0, 1].set_ylabel('Predicted RUL')
axes[0, 1].grid(True, alpha=0.3)

errors = y_val - y_val_pred
axes[1, 0].hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title(f'Error Distribution (MAE={val_mae:.1f})', fontweight='bold')
axes[1, 0].set_xlabel('Error')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

sample = np.random.choice(len(y_val), min(100, len(y_val)), replace=False)
axes[1, 1].plot(y_val[sample], label='Actual', marker='o', linewidth=2, markersize=4)
axes[1, 1].plot(y_val_pred[sample], label='Predicted', marker='s', linewidth=2, markersize=4)
axes[1, 1].set_title('Sample Predictions', fontweight='bold')
axes[1, 1].set_xlabel('Sample')
axes[1, 1].set_ylabel('RUL')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/diagrams/eda/final_lstm_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: docs/diagrams/eda/final_lstm_results.png")
plt.close()

# Save metrics
metrics = {
    'val_rmse': float(val_rmse),
    'val_mae': float(val_mae),
    'val_r2': float(val_r2),
    'features': feature_cols
}

with open(models_dir / 'final_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🎯 FINAL MODEL PERFORMANCE:")
print(f"   Validation RMSE: {val_rmse:.2f} cycles")
print(f"   Validation MAE:  {val_mae:.2f} cycles")
print(f"   R² Score:        {val_r2:.4f}")
print(f"\n📁 Model saved: {models_dir}/best_model_top_sensors.keras")
print("=" * 70)
