"""
Load and re-export model in correct .h5 format
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

from tensorflow import keras
import pickle

print("=" * 70)
print("RE-EXPORTING MODEL IN CORRECT FORMAT")
print("=" * 70)

# Load the model from trained directory
source_path = 'models/trained/best_model_top_sensors.h5'
dest_path = 'ml_models/best_model_top_sensors.h5'

print(f"\n📂 Loading model from: {source_path}")

try:
    # Load model
    model = keras.models.load_model(source_path, compile=True)
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Re-save with explicit format
    print(f"\n💾 Saving to: {dest_path}")
    model.save(dest_path, save_format='h5', overwrite=True)
    print(f"✅ Model saved successfully")
    
    # Test reload
    print(f"\n🧪 Testing reload...")
    test_model = keras.models.load_model(dest_path)
    print(f"✅ Model reloads correctly!")
    
    # Copy scaler
    import shutil
    scaler_source = 'models/scaler_top_sensors.pkl'
    scaler_dest = 'ml_models/scaler_top_sensors.pkl'
    
    if os.path.exists(scaler_source):
        shutil.copy2(scaler_source, scaler_dest)
        print(f"\n✅ Scaler copied: {scaler_dest}")
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\nFiles ready:")
    print(f"  - {dest_path}")
    print(f"  - {scaler_dest}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

