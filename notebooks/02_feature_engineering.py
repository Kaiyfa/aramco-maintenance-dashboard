"""
Feature Engineering for Predictive Maintenance
Creates rolling statistics, degradation indicators, and time-series features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FEATURE ENGINEERING FOR PREDICTIVE MAINTENANCE")
print("=" * 70)

# Load processed data with RUL
data_path = Path('data/processed/train_FD001_with_RUL.csv')
df = pd.read_csv(data_path)

print(f"\n✅ Loaded data: {df.shape}")
print(f"Engines: {df['unit_id'].nunique()}")

# Load useful sensors
with open('data/processed/useful_sensors.txt', 'r') as f:
    useful_sensors = [line.strip() for line in f]

print(f"Useful sensors: {len(useful_sensors)}")

# ============================================================
# 1. ROLLING STATISTICS (Capture trends)
# ============================================================
print("\n" + "=" * 70)
print("1. CREATING ROLLING STATISTICS FEATURES")
print("=" * 70)

window_sizes = [5, 10, 20]
rolling_features = []

for sensor in useful_sensors:
    for window in window_sizes:
        # Rolling mean
        col_name_mean = f'{sensor}_rolling_mean_{window}'
        df[col_name_mean] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        rolling_features.append(col_name_mean)
        
        # Rolling std (volatility)
        col_name_std = f'{sensor}_rolling_std_{window}'
        df[col_name_std] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        rolling_features.append(col_name_std)

print(f"✅ Created {len(rolling_features)} rolling features")

# ============================================================
# 2. DEGRADATION INDICATORS (Rate of change)
# ============================================================
print("\n" + "=" * 70)
print("2. CREATING DEGRADATION INDICATORS")
print("=" * 70)

degradation_features = []

for sensor in useful_sensors:
    # Difference from previous cycle (rate of degradation)
    col_name_diff = f'{sensor}_diff'
    df[col_name_diff] = df.groupby('unit_id')[sensor].diff()
    degradation_features.append(col_name_diff)
    
    # Cumulative sum (total degradation)
    col_name_cumsum = f'{sensor}_cumsum'
    df[col_name_cumsum] = df.groupby('unit_id')[sensor].cumsum()
    degradation_features.append(col_name_cumsum)

# Fill NaN values from diff operation
df[degradation_features] = df[degradation_features].fillna(0)

print(f"✅ Created {len(degradation_features)} degradation indicators")

# ============================================================
# 3. EXPONENTIAL WEIGHTED FEATURES (Recent trends matter more)
# ============================================================
print("\n" + "=" * 70)
print("3. CREATING EXPONENTIAL WEIGHTED FEATURES")
print("=" * 70)

ewm_features = []
alpha_values = [0.1, 0.3, 0.5]

for sensor in useful_sensors:
    for alpha in alpha_values:
        col_name = f'{sensor}_ewm_{int(alpha*10)}'
        df[col_name] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.ewm(alpha=alpha).mean()
        )
        ewm_features.append(col_name)

print(f"✅ Created {len(ewm_features)} exponential weighted features")

# ============================================================
# 4. TIME-BASED FEATURES
# ============================================================
print("\n" + "=" * 70)
print("4. CREATING TIME-BASED FEATURES")
print("=" * 70)

# Cycle percentage (how far into lifecycle)
df['cycle_percentage'] = df['time_cycle'] / df['max_cycle']

# Log of time cycle (captures non-linear degradation)
df['time_cycle_log'] = np.log1p(df['time_cycle'])

# Polynomial features for time
df['time_cycle_squared'] = df['time_cycle'] ** 2
df['time_cycle_cubed'] = df['time_cycle'] ** 3

time_features = ['cycle_percentage', 'time_cycle_log', 
                 'time_cycle_squared', 'time_cycle_cubed']

print(f"✅ Created {len(time_features)} time-based features")

# ============================================================
# 5. CLIP RUL (Piece-wise linear degradation assumption)
# ============================================================
print("\n" + "=" * 70)
print("5. APPLYING RUL CLIPPING STRATEGY")
print("=" * 70)

# Common practice: Equipment is "healthy" until certain threshold
# After threshold, degradation becomes predictable
RUL_THRESHOLD = 125

df['RUL_clipped'] = df['RUL'].clip(upper=RUL_THRESHOLD)

print(f"✅ RUL clipped at {RUL_THRESHOLD} cycles")
print(f"   Original RUL max: {df['RUL'].max()}")
print(f"   Clipped RUL max: {df['RUL_clipped'].max()}")

# ============================================================
# 6. FEATURE SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

all_feature_cols = (
    useful_sensors + 
    rolling_features + 
    degradation_features + 
    ewm_features + 
    time_features +
    ['op_setting_1', 'op_setting_2', 'op_setting_3']
)

print(f"Original sensors: {len(useful_sensors)}")
print(f"Rolling features: {len(rolling_features)}")
print(f"Degradation features: {len(degradation_features)}")
print(f"EWM features: {len(ewm_features)}")
print(f"Time features: {len(time_features)}")
print(f"Operational settings: 3")
print(f"\n{'='*70}")
print(f"TOTAL FEATURES: {len(all_feature_cols)}")
print(f"{'='*70}")

# ============================================================
# 7. NORMALIZE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("7. NORMALIZING FEATURES (MinMax Scaling)")
print("=" * 70)

scaler = MinMaxScaler()
df[all_feature_cols] = scaler.fit_transform(df[all_feature_cols])

print("✅ All features normalized to [0, 1] range")

# ============================================================
# 8. SAVE ENGINEERED FEATURES
# ============================================================
print("\n" + "=" * 70)
print("8. SAVING ENGINEERED DATASET")
print("=" * 70)

# Save full engineered dataset
output_path = Path('data/processed/train_FD001_engineered.csv')
df.to_csv(output_path, index=False)
print(f"✅ Saved: {output_path}")
print(f"   Shape: {df.shape}")

# Save feature list
feature_list_path = Path('data/processed/feature_list.txt')
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(all_feature_cols))
print(f"✅ Saved: {feature_list_path}")

# Save scaler for later use
import joblib
scaler_path = Path('models/scaler.pkl')
scaler_path.parent.mkdir(exist_ok=True)
joblib.dump(scaler, scaler_path)
print(f"✅ Saved: {scaler_path}")

# ============================================================
# 9. SAMPLE DATA VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("9. SAMPLE ENGINEERED DATA")
print("=" * 70)

sample_engine = df[df['unit_id'] == 1][['time_cycle', 'RUL', 'RUL_clipped'] + 
                                        useful_sensors[:5] + 
                                        rolling_features[:3]].head(10)
print(sample_engine)

print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print("\nReady for model training with:")
print(f"  • {len(all_feature_cols)} features")
print(f"  • {len(df)} data points")
print(f"  • {df['unit_id'].nunique()} engines")
print("=" * 70)
