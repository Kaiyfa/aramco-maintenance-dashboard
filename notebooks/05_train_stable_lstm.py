"""
Stable LSTM Training with Gradient Clipping
Addresses NaN prediction issue
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("STABLE LSTM MODEL TRAINING")
print("=" * 70)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "=" * 70)
print("1. LOADING SEQUENCES")
print("=" * 70)

data_dir = Path('data/processed/sequences')

X_train = np.load(data_dir / 'X_train.npy')
y_train = np.load(data_dir / 'y_train.npy')
X_val = np.load(data_dir / 'X_val.npy')
y_val = np.load(data_dir / 'y_val.npy')

print(f"✅ Data loaded: {X_train.shape}, {X_val.shape}")

# ============================================================
# 2. BUILD SIMPLER, MORE STABLE MODEL
# ============================================================
print("\n" + "=" * 70)
print("2. BUILDING STABLE LSTM ARCHITECTURE")
print("=" * 70)

model = Sequential([
    # Single LSTM layer (simpler = more stable)
    LSTM(64, return_sequences=True, 
         input_shape=(X_train.shape[1], X_train.shape[2]),
         kernel_initializer='glorot_uniform'),
    Dropout(0.3),
    
    # Second LSTM layer
    LSTM(32, return_sequences=False,
         kernel_initializer='glorot_uniform'),
    Dropout(0.3),
    
    # Dense output layers
    Dense(16, activation='relu'),
    Dense(1, activation='relu')  # ReLU ensures positive RUL
])

print(model.summary())

# ============================================================
# 3. COMPILE WITH GRADIENT CLIPPING
# ============================================================
print("\n" + "=" * 70)
print("3. COMPILING WITH GRADIENT CLIPPING")
print("=" * 70)

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Gradient clipping!

model.compile(
    optimizer=optimizer,
    loss='huber',  # More robust to outliers than MSE
    metrics=['mae']
)

print("✅ Model compiled with:")
print("   • Gradient clipping (clipnorm=1.0)")
print("   • Huber loss (robust to outliers)")

# ============================================================
# 4. TRAIN MODEL
# ============================================================
print("\n" + "=" * 70)
print("4. TRAINING MODEL")
print("=" * 70)

models_dir = Path('models/trained')
models_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(models_dir / 'stable_lstm_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print("🚀 Training started...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")

# ============================================================
# 5. EVALUATE MODEL
# ============================================================
print("\n" + "=" * 70)
print("5. EVALUATING MODEL")
print("=" * 70)

# Make predictions
y_train_pred = model.predict(X_train, batch_size=256, verbose=0).flatten()
y_val_pred = model.predict(X_val, batch_size=256, verbose=0).flatten()

# Check for NaN
train_nan = np.isnan(y_train_pred).sum()
val_nan = np.isnan(y_val_pred).sum()

print(f"NaN check:")
print(f"   Training NaN count: {train_nan}")
print(f"   Validation NaN count: {val_nan}")

if train_nan > 0 or val_nan > 0:
    print("\n⚠️  WARNING: Still have NaN values!")
    print("   Model needs further debugging")
else:
    print("\n✅ No NaN values - predictions are valid!")
    
    # Clip predictions
    y_train_pred = np.clip(y_train_pred, 0, 125)
    y_val_pred = np.clip(y_val_pred, 0, 125)
    
    # Calculate metrics
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
    
    # Performance assessment
    if val_rmse < 20:
        print("\n🌟 EXCELLENT PERFORMANCE!")
    elif val_rmse < 30:
        print("\n✅ GOOD PERFORMANCE!")
    else:
        print("\n⚠️  Performance needs improvement")
    
    # ========================================================
    # 6. VISUALIZATIONS
    # ========================================================
    print("\n" + "=" * 70)
    print("6. CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    axes[0, 0].plot(history.history['loss'], label='Training', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Training History', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation predictions
    axes[0, 1].scatter(y_val, y_val_pred, alpha=0.4, s=15)
    axes[0, 1].plot([0, 125], [0, 125], 'r--', linewidth=2)
    axes[0, 1].set_title(f'Validation Predictions (R²={val_r2:.3f})', 
                         fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Actual RUL')
    axes[0, 1].set_ylabel('Predicted RUL')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    errors = y_val - y_val_pred
    axes[1, 0].hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Prediction Errors', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Error (Actual - Predicted)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample predictions
    sample_idx = np.random.choice(len(y_val), 100, replace=False)
    axes[1, 1].plot(y_val[sample_idx], label='Actual', marker='o', linewidth=2, markersize=4)
    axes[1, 1].plot(y_val_pred[sample_idx], label='Predicted', marker='s', linewidth=2, markersize=4)
    axes[1, 1].set_title('Sample Predictions', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('RUL (cycles)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/diagrams/eda/stable_lstm_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: docs/diagrams/eda/stable_lstm_results.png")
    plt.close()
    
    # ========================================================
    # 7. SAVE RESULTS
    # ========================================================
    metrics = {
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2)
    }
    
    with open(models_dir / 'stable_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"✅ Saved metrics")

print("\n" + "=" * 70)
print("✅ STABLE MODEL TRAINING COMPLETE!")
print("=" * 70)
