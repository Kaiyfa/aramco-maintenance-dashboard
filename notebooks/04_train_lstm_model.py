"""
Train LSTM Model for Remaining Useful Life (RUL) Prediction
Deep learning model for predictive maintenance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("LSTM MODEL TRAINING FOR PREDICTIVE MAINTENANCE")
print("=" * 70)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"\n✅ TensorFlow version: {tf.__version__}")
print(f"✅ GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ============================================================
# 1. LOAD PREPARED SEQUENCES
# ============================================================
print("\n" + "=" * 70)
print("1. LOADING PREPARED SEQUENCES")
print("=" * 70)

data_dir = Path('data/processed/sequences')

X_train = np.load(data_dir / 'X_train.npy')
y_train = np.load(data_dir / 'y_train.npy')
X_val = np.load(data_dir / 'X_val.npy')
y_val = np.load(data_dir / 'y_val.npy')

with open(data_dir / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"✅ Training data loaded:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"✅ Validation data loaded:")
print(f"   X_val: {X_val.shape}")
print(f"   y_val: {y_val.shape}")

sequence_length = metadata['sequence_length']
n_features = metadata['n_features']

print(f"\nSequence length: {sequence_length}")
print(f"Number of features: {n_features}")

# ============================================================
# 2. BUILD LSTM MODEL
# ============================================================
print("\n" + "=" * 70)
print("2. BUILDING LSTM ARCHITECTURE")
print("=" * 70)

model = Sequential([
    # First LSTM layer with return sequences
    LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    BatchNormalization(),
    
    # Second LSTM layer with return sequences
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    
    # Third LSTM layer (final recurrent layer)
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),
    
    # Dense layers for prediction
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1)  # Output: RUL prediction
])

print("✅ Model architecture created!")
print("\nModel Summary:")
print(model.summary())

# ============================================================
# 3. COMPILE MODEL
# ============================================================
print("\n" + "=" * 70)
print("3. COMPILING MODEL")
print("=" * 70)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

print("✅ Model compiled!")
print("   Optimizer: Adam (lr=0.001)")
print("   Loss: Mean Squared Error")
print("   Metrics: Mean Absolute Error")

# ============================================================
# 4. SETUP CALLBACKS
# ============================================================
print("\n" + "=" * 70)
print("4. SETTING UP TRAINING CALLBACKS")
print("=" * 70)

# Create models directory
models_dir = Path('models/trained')
models_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        filepath=str(models_dir / 'best_lstm_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # Reduce learning rate when plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

print("✅ Callbacks configured:")
print("   • Early Stopping (patience=15)")
print("   • Model Checkpoint (save best)")
print("   • Learning Rate Reduction")

# ============================================================
# 5. TRAIN MODEL
# ============================================================
print("\n" + "=" * 70)
print("5. TRAINING LSTM MODEL")
print("=" * 70)
print("\n🚀 Starting training...\n")

start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

print(f"\n✅ Training completed in {training_time/60:.2f} minutes!")

# ============================================================
# 6. EVALUATE MODEL
# ============================================================
print("\n" + "=" * 70)
print("6. EVALUATING MODEL PERFORMANCE")
print("=" * 70)

# Predictions
y_train_pred = model.predict(X_train, verbose=0).flatten()
y_val_pred = model.predict(X_val, verbose=0).flatten()

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("\n📊 TRAINING SET PERFORMANCE:")
print(f"   RMSE: {train_rmse:.2f} cycles")
print(f"   MAE:  {train_mae:.2f} cycles")
print(f"   R²:   {train_r2:.4f}")

print("\n📊 VALIDATION SET PERFORMANCE:")
print(f"   RMSE: {val_rmse:.2f} cycles")
print(f"   MAE:  {val_mae:.2f} cycles")
print(f"   R²:   {val_r2:.4f}")

# ============================================================
# 7. VISUALIZE TRAINING HISTORY
# ============================================================
print("\n" + "=" * 70)
print("7. GENERATING TRAINING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss curves
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Model Loss Over Epochs', fontweight='bold', fontsize=14)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MAE curves
axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_title('Mean Absolute Error Over Epochs', fontweight='bold', fontsize=14)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Training predictions scatter
axes[1, 0].scatter(y_train, y_train_pred, alpha=0.3, s=10)
axes[1, 0].plot([0, 125], [0, 125], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_title(f'Training Set Predictions (R²={train_r2:.4f})', fontweight='bold', fontsize=14)
axes[1, 0].set_xlabel('Actual RUL (cycles)')
axes[1, 0].set_ylabel('Predicted RUL (cycles)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Validation predictions scatter
axes[1, 1].scatter(y_val, y_val_pred, alpha=0.3, s=10, color='orange')
axes[1, 1].plot([0, 125], [0, 125], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_title(f'Validation Set Predictions (R²={val_r2:.4f})', fontweight='bold', fontsize=14)
axes[1, 1].set_xlabel('Actual RUL (cycles)')
axes[1, 1].set_ylabel('Predicted RUL (cycles)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/diagrams/eda/lstm_training_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: docs/diagrams/eda/lstm_training_results.png")
plt.close()

# ============================================================
# 8. SAVE FINAL MODEL
# ============================================================
print("\n" + "=" * 70)
print("8. SAVING TRAINED MODEL")
print("=" * 70)

# Save in Keras format
model.save(models_dir / 'final_lstm_model.keras')
print(f"✅ Saved: {models_dir}/final_lstm_model.keras")

# Save training history
history_path = models_dir / 'training_history.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"✅ Saved: {history_path}")

# Save metrics
metrics = {
    'train_rmse': float(train_rmse),
    'train_mae': float(train_mae),
    'train_r2': float(train_r2),
    'val_rmse': float(val_rmse),
    'val_mae': float(val_mae),
    'val_r2': float(val_r2),
    'training_time_minutes': training_time / 60
}

metrics_path = models_dir / 'metrics.pkl'
with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)
print(f"✅ Saved: {metrics_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✅ LSTM MODEL TRAINING COMPLETE!")
print("=" * 70)
print("\n🎯 FINAL PERFORMANCE METRICS:")
print(f"   Validation RMSE: {val_rmse:.2f} cycles")
print(f"   Validation MAE:  {val_mae:.2f} cycles")
print(f"   Validation R²:   {val_r2:.4f}")
print(f"\n⏱️  Training time: {training_time/60:.2f} minutes")
print("\n📁 Saved artifacts:")
print(f"   • Model: {models_dir}/final_lstm_model.keras")
print(f"   • Best checkpoint: {models_dir}/best_lstm_model.keras")
print(f"   • Training history: {models_dir}/training_history.pkl")
print(f"   • Metrics: {models_dir}/metrics.pkl")
print(f"   • Visualization: docs/diagrams/eda/lstm_training_results.png")
print("=" * 70)
