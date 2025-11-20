"""
Exploratory Data Analysis for NASA CMAPSS Dataset
Focus: Understanding sensor patterns and equipment degradation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("NASA CMAPSS FD001 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Define column names
column_names = [
    'unit_id', 'time_cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Load training data
train_path = Path('data/raw/cmapss/train_FD001.txt')
train_df = pd.read_csv(train_path, sep='\s+', header=None, names=column_names)

print("\n✅ Data loaded successfully!")
print(f"Shape: {train_df.shape}")

# ============================================================
# 1. CALCULATE REMAINING USEFUL LIFE (RUL)
# ============================================================
print("\n" + "=" * 70)
print("1. CALCULATING REMAINING USEFUL LIFE (RUL)")
print("=" * 70)

# Get max cycle for each engine (failure point)
max_cycles = train_df.groupby('unit_id')['time_cycle'].max().reset_index()
max_cycles.columns = ['unit_id', 'max_cycle']

# Merge and calculate RUL
train_df = train_df.merge(max_cycles, on='unit_id', how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycle']

print(f"✅ RUL calculated for all {train_df['unit_id'].nunique()} engines")
print(f"\nRUL Statistics:")
print(f"  Mean RUL: {train_df['RUL'].mean():.1f} cycles")
print(f"  Max RUL: {train_df['RUL'].max()} cycles")
print(f"  Min RUL: {train_df['RUL'].min()} cycles")

# ============================================================
# 2. IDENTIFY CONSTANT SENSORS (No predictive value)
# ============================================================
print("\n" + "=" * 70)
print("2. IDENTIFYING CONSTANT SENSORS")
print("=" * 70)

sensor_cols = [col for col in column_names if 'sensor' in col]
constant_sensors = []

for sensor in sensor_cols:
    if train_df[sensor].std() < 0.01:  # Very low variance
        constant_sensors.append(sensor)
        print(f"❌ {sensor}: Constant (std={train_df[sensor].std():.6f})")

useful_sensors = [s for s in sensor_cols if s not in constant_sensors]
print(f"\n✅ {len(useful_sensors)} useful sensors identified")
print(f"❌ {len(constant_sensors)} constant sensors removed")

# ============================================================
# 3. SENSOR DEGRADATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("3. ANALYZING SENSOR DEGRADATION PATTERNS")
print("=" * 70)

# Select 5 engines for detailed analysis
sample_engines = [1, 2, 3, 4, 5]

# Create visualization directory
import os
os.makedirs('docs/diagrams/eda', exist_ok=True)

# Plot sensor trends over lifecycle
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle('Sensor Degradation Patterns (First 5 Engines)', fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, sensor in enumerate(useful_sensors[:12]):  # Plot first 12 useful sensors
    ax = axes[idx]
    
    for engine_id in sample_engines:
        engine_data = train_df[train_df['unit_id'] == engine_id]
        ax.plot(engine_data['time_cycle'], engine_data[sensor], 
                alpha=0.7, linewidth=2, label=f'Engine {engine_id}')
    
    ax.set_title(f'{sensor}', fontweight='bold')
    ax.set_xlabel('Time Cycle')
    ax.set_ylabel('Sensor Value')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('docs/diagrams/eda/sensor_degradation_patterns.png', dpi=300, bbox_inches='tight')
print("✅ Saved: docs/diagrams/eda/sensor_degradation_patterns.png")
plt.close()

# ============================================================
# 4. RUL DISTRIBUTION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("4. RUL DISTRIBUTION ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(train_df['RUL'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('RUL Distribution Across All Measurements', fontweight='bold')
axes[0].set_xlabel('Remaining Useful Life (cycles)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(train_df['RUL'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {train_df["RUL"].mean():.1f}')
axes[0].legend()

# Engine lifecycle distribution
lifecycle_dist = max_cycles['max_cycle']
axes[1].hist(lifecycle_dist, bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_title('Engine Lifecycle Distribution', fontweight='bold')
axes[1].set_xlabel('Total Lifecycle (cycles)')
axes[1].set_ylabel('Number of Engines')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(lifecycle_dist.mean(), color='darkred', linestyle='--', 
                linewidth=2, label=f'Mean: {lifecycle_dist.mean():.1f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('docs/diagrams/eda/rul_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: docs/diagrams/eda/rul_distribution.png")
plt.close()

# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("5. SENSOR CORRELATION WITH RUL")
print("=" * 70)

# Calculate correlation with RUL
correlations = train_df[useful_sensors + ['RUL']].corr()['RUL'].drop('RUL').sort_values(ascending=False)

print("\nTop 10 sensors most correlated with RUL:")
print(correlations.head(10))

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))
correlations.plot(kind='barh', ax=ax, color='teal', edgecolor='black')
ax.set_title('Sensor Correlation with Remaining Useful Life (RUL)', fontweight='bold', fontsize=14)
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('Sensors')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('docs/diagrams/eda/sensor_rul_correlation.png', dpi=300, bbox_inches='tight')
print("✅ Saved: docs/diagrams/eda/sensor_rul_correlation.png")
plt.close()

# ============================================================
# 6. HEATMAP OF SENSOR CORRELATIONS
# ============================================================
print("\n" + "=" * 70)
print("6. SENSOR INTERCORRELATION HEATMAP")
print("=" * 70)

# Select top correlated sensors for cleaner visualization
top_sensors = correlations.abs().head(10).index.tolist()
correlation_matrix = train_df[top_sensors].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Intercorrelation Between Top 10 Predictive Sensors', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('docs/diagrams/eda/sensor_intercorrelation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: docs/diagrams/eda/sensor_intercorrelation_heatmap.png")
plt.close()

# ============================================================
# 7. SAVE PROCESSED DATA
# ============================================================
print("\n" + "=" * 70)
print("7. SAVING PROCESSED DATA")
print("=" * 70)

# Save cleaned data with RUL
processed_path = Path('data/processed/train_FD001_with_RUL.csv')
processed_path.parent.mkdir(exist_ok=True)
train_df.to_csv(processed_path, index=False)
print(f"✅ Saved: {processed_path}")

# Save list of useful sensors
useful_sensors_path = Path('data/processed/useful_sensors.txt')
with open(useful_sensors_path, 'w') as f:
    f.write('\n'.join(useful_sensors))
print(f"✅ Saved: {useful_sensors_path}")

# ============================================================
# SUMMARY REPORT
# ============================================================
print("\n" + "=" * 70)
print("EDA SUMMARY REPORT")
print("=" * 70)
print(f"Total Data Points: {len(train_df):,}")
print(f"Number of Engines: {train_df['unit_id'].nunique()}")
print(f"Average Lifecycle: {max_cycles['max_cycle'].mean():.1f} cycles")
print(f"Useful Sensors: {len(useful_sensors)} out of {len(sensor_cols)}")
print(f"Constant Sensors Removed: {len(constant_sensors)}")
print(f"\nMost Predictive Sensors:")
for i, (sensor, corr) in enumerate(correlations.head(5).items(), 1):
    print(f"  {i}. {sensor}: {corr:.4f}")

print("\n" + "=" * 70)
print("✅ EXPLORATORY DATA ANALYSIS COMPLETE!")
print("=" * 70)
print("\nGenerated Visualizations:")
print("  1. docs/diagrams/eda/sensor_degradation_patterns.png")
print("  2. docs/diagrams/eda/rul_distribution.png")
print("  3. docs/diagrams/eda/sensor_rul_correlation.png")
print("  4. docs/diagrams/eda/sensor_intercorrelation_heatmap.png")
print("\nProcessed Data:")
print("  1. data/processed/train_FD001_with_RUL.csv")
print("  2. data/processed/useful_sensors.txt")
print("=" * 70)
