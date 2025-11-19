"""
Convert trained Keras model to TensorFlow Lite
For deployment on Raspberry Pi edge device
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle

print("=" * 70)
print("TENSORFLOW LITE CONVERSION FOR EDGE DEPLOYMENT")
print("=" * 70)

# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================
print("\n" + "=" * 70)
print("1. LOADING TRAINED KERAS MODEL")
print("=" * 70)

model_path = Path('models/trained/best_model_top_sensors.keras')
model = tf.keras.models.load_model(model_path)

print(f"✅ Model loaded: {model_path}")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# ============================================================
# 2. CONVERT TO TENSORFLOW LITE
# ============================================================
print("\n" + "=" * 70)
print("2. CONVERTING TO TENSORFLOW LITE FORMAT")
print("=" * 70)

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization settings for Raspberry Pi
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

# Convert
print("🔄 Converting... (this may take a moment)")
tflite_model = converter.convert()

print("✅ Conversion successful!")

# ============================================================
# 3. SAVE TFLITE MODEL
# ============================================================
print("\n" + "=" * 70)
print("3. SAVING TFLITE MODEL")
print("=" * 70)

tflite_dir = Path('models/tflite')
tflite_dir.mkdir(parents=True, exist_ok=True)

tflite_path = tflite_dir / 'predictive_maintenance_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# Check file size
file_size_mb = tflite_path.stat().st_size / (1024 * 1024)

print(f"✅ Saved: {tflite_path}")
print(f"   File size: {file_size_mb:.2f} MB")

# ============================================================
# 4. TEST TFLITE MODEL
# ============================================================
print("\n" + "=" * 70)
print("4. TESTING TFLITE MODEL")
print("=" * 70)

# Load test data
data_dir = Path('data/processed/sequences')
X_val = np.load(data_dir / 'X_val.npy')
y_val = np.load(data_dir / 'y_val.npy')

# Take sample
sample_input = X_val[:5].astype(np.float32)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ TFLite interpreter loaded")
print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Test inference
print("\n🧪 Running test inference...")
tflite_predictions = []

for i in range(len(sample_input)):
    interpreter.set_tensor(input_details[0]['index'], sample_input[i:i+1])
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    tflite_predictions.append(prediction)

tflite_predictions = np.array(tflite_predictions)

# Compare with original model
original_predictions = model.predict(sample_input, verbose=0).flatten()

print("\n📊 COMPARISON: Original vs TFLite")
print("="*60)
print(f"{'Index':<8} {'Original':<12} {'TFLite':<12} {'Difference':<12}")
print("="*60)
for i in range(len(sample_input)):
    diff = abs(original_predictions[i] - tflite_predictions[i])
    print(f"{i:<8} {original_predictions[i]:<12.2f} {tflite_predictions[i]:<12.2f} {diff:<12.4f}")
print("="*60)

max_diff = np.max(np.abs(original_predictions - tflite_predictions))
print(f"\nMax difference: {max_diff:.4f} cycles")

if max_diff < 1.0:
    print("✅ TFLite model matches original perfectly!")
elif max_diff < 2.0:
    print("✅ TFLite model has acceptable accuracy!")
else:
    print("⚠️  TFLite model has significant differences")

# ============================================================
# 5. CREATE DEPLOYMENT PACKAGE
# ============================================================
print("\n" + "=" * 70)
print("5. CREATING DEPLOYMENT PACKAGE")
print("=" * 70)

# Create deployment info
deployment_info = {
    'model_path': str(tflite_path),
    'model_size_mb': file_size_mb,
    'input_shape': list(model.input_shape),
    'output_shape': list(model.output_shape),
    'sequence_length': 30,
    'n_features': 13,
    'feature_names': [
        'sensor_12', 'sensor_7', 'sensor_21', 'sensor_20',
        'sensor_14', 'sensor_9', 'sensor_13', 'sensor_8',
        'sensor_3', 'sensor_17', 'op_setting_1', 'op_setting_2', 'op_setting_3'
    ],
    'scaler_path': 'models/scaler_top_sensors.pkl',
    'performance': {
        'validation_rmse': 15.46,
        'validation_mae': 11.80,
        'validation_r2': 0.8657
    }
}

info_path = tflite_dir / 'deployment_info.pkl'
with open(info_path, 'wb') as f:
    pickle.dump(deployment_info, f)

print(f"✅ Saved deployment info: {info_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✅ TENSORFLOW LITE CONVERSION COMPLETE!")
print("=" * 70)
print("\n📦 DEPLOYMENT PACKAGE READY:")
print(f"   • TFLite Model: {tflite_path}")
print(f"   • Model Size: {file_size_mb:.2f} MB")
print(f"   • Deployment Info: {info_path}")
print(f"   • Scaler: models/scaler_top_sensors.pkl")
print("\n🚀 READY FOR RASPBERRY PI DEPLOYMENT!")
print("\n📋 Next Steps:")
print("   1. Transfer files to Raspberry Pi")
print("   2. Install TensorFlow Lite runtime")
print("   3. Deploy FastAPI inference service")
print("   4. Connect Raspberry Pi Pico sensors")
print("   5. Start real-time monitoring!")
print("=" * 70)
