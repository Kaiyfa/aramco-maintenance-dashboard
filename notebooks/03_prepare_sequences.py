"""
Prepare Time-Series Sequences for LSTM Training
Creates sliding windows of sensor data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

print("=" * 70)
print("PREPARING TIME-SERIES SEQUENCES FOR LSTM")
print("=" * 70)

# Load engineered data
data_path = Path('data/processed/train_FD001_engineered.csv')
df = pd.read_csv(data_path)

# Load feature list
with open('data/processed/feature_list.txt', 'r') as f:
    feature_cols = [line.strip() for line in f]

print(f"\n✅ Loaded engineered data: {df.shape}")
print(f"Features: {len(feature_cols)}")

# ============================================================
# 1. SEQUENCE PARAMETERS
# ============================================================
SEQUENCE_LENGTH = 30  # Use last 30 cycles to predict RUL
print(f"\n{'='*70}")
print(f"Sequence length: {SEQUENCE_LENGTH} time steps")
print(f"{'='*70}")

# ============================================================
# 2. CREATE SEQUENCES FUNCTION
# ============================================================
def create_sequences(data, unit_col, feature_cols, target_col, seq_length):
    """
    Create sliding window sequences for LSTM
    
    Args:
        data: DataFrame with all data
        unit_col: Column name for unit/engine ID
        feature_cols: List of feature column names
        target_col: Target variable (RUL)
        seq_length: Length of sequence window
    
    Returns:
        X: Array of shape (samples, seq_length, n_features)
        y: Array of shape (samples,)
    """
    sequences = []
    targets = []
    
    # Process each engine separately
    for unit_id in data[unit_col].unique():
        unit_data = data[data[unit_col] == unit_id]
        
        # Get features and target
        features = unit_data[feature_cols].values
        target = unit_data[target_col].values
        
        # Create sequences using sliding window
        for i in range(len(features) - seq_length + 1):
            seq = features[i:i + seq_length]
            label = target[i + seq_length - 1]  # Predict RUL at end of sequence
            
            sequences.append(seq)
            targets.append(label)
    
    return np.array(sequences), np.array(targets)

# ============================================================
# 3. GENERATE SEQUENCES
# ============================================================
print("\n" + "=" * 70)
print("GENERATING SEQUENCES...")
print("=" * 70)

X, y = create_sequences(
    data=df,
    unit_col='unit_id',
    feature_cols=feature_cols,
    target_col='RUL_clipped',
    seq_length=SEQUENCE_LENGTH
)

print(f"✅ Sequences created!")
print(f"\nX shape: {X.shape}")
print(f"  → {X.shape[0]} sequences")
print(f"  → {X.shape[1]} time steps per sequence")
print(f"  → {X.shape[2]} features per time step")
print(f"\ny shape: {y.shape}")
print(f"  → {y.shape[0]} target values (RUL)")

# ============================================================
# 4. TRAIN/VALIDATION SPLIT
# ============================================================
print("\n" + "=" * 70)
print("SPLITTING DATA: TRAIN / VALIDATION")
print("=" * 70)

# Use 80% for training, 20% for validation
split_idx = int(0.8 * len(X))

X_train = X[:split_idx]
y_train = y[:split_idx]
X_val = X[split_idx:]
y_val = y[split_idx:]

print(f"Training set:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"\nValidation set:")
print(f"  X_val: {X_val.shape}")
print(f"  y_val: {y_val.shape}")

# ============================================================
# 5. SAVE SEQUENCES
# ============================================================
print("\n" + "=" * 70)
print("SAVING PREPARED SEQUENCES")
print("=" * 70)

output_dir = Path('data/processed/sequences')
output_dir.mkdir(exist_ok=True)

# Save as numpy arrays
np.save(output_dir / 'X_train.npy', X_train)
np.save(output_dir / 'y_train.npy', y_train)
np.save(output_dir / 'X_val.npy', X_val)
np.save(output_dir / 'y_val.npy', y_val)

print("✅ Saved training sequences:")
print(f"   • {output_dir}/X_train.npy")
print(f"   • {output_dir}/y_train.npy")
print("✅ Saved validation sequences:")
print(f"   • {output_dir}/X_val.npy")
print(f"   • {output_dir}/y_val.npy")

# Save metadata
metadata = {
    'sequence_length': SEQUENCE_LENGTH,
    'n_features': len(feature_cols),
    'n_train_samples': len(X_train),
    'n_val_samples': len(X_val),
    'feature_cols': feature_cols
}

with open(output_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"✅ Saved metadata: {output_dir}/metadata.pkl")

# ============================================================
# 6. DATA SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SEQUENCE PREPARATION SUMMARY")
print("=" * 70)
print(f"Sequence length: {SEQUENCE_LENGTH} time steps")
print(f"Features per time step: {len(feature_cols)}")
print(f"Total sequences: {len(X):,}")
print(f"Training sequences: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation sequences: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"\nTarget (RUL) statistics:")
print(f"  Mean: {y.mean():.2f} cycles")
print(f"  Std: {y.std():.2f} cycles")
print(f"  Min: {y.min():.2f} cycles")
print(f"  Max: {y.max():.2f} cycles")

print("\n" + "=" * 70)
print("✅ SEQUENCE PREPARATION COMPLETE!")
print("=" * 70)
print("\nReady for LSTM model training!")
