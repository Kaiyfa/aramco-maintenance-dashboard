"""
ML Prediction Service - Production Ready (FIXED)
Handles LSTM model loading and RUL predictions with 30-cycle sequences
"""

import os
import joblib
import numpy as np
from tensorflow import keras
import logging
from typing import Dict, List, Tuple
import time

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Production-grade ML prediction service for equipment RUL prediction
    Uses LSTM model trained on NASA C-MAPSS dataset
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.sequence_length = 30  # LSTM requires 30 timesteps
        self.n_features = 13  # 13 sensor features
        
    def load_model(self) -> bool:
        """Load LSTM model and scaler from disk"""
        try:
            model_path = os.path.join('ml_models', 'best_model_top_sensors.h5')
            scaler_path = os.path.join('ml_models', 'scaler_top_sensors.pkl')
            
            # Load Keras model
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                logger.info(f"✅ Model loaded: {model_path}")
                logger.info(f"   Input shape: {self.model.input_shape}")
                logger.info(f"   Output shape: {self.model.output_shape}")
                
                # Validate model architecture
                if len(self.model.input_shape) == 3:
                    self.sequence_length = self.model.input_shape[1]
                    self.n_features = self.model.input_shape[2]
                    logger.info(f"   Sequence length: {self.sequence_length}")
                    logger.info(f"   Features: {self.n_features}")
            else:
                logger.error(f"❌ Model not found: {model_path}")
                return False
            
            # Load StandardScaler
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler loaded: {scaler_path}")
                logger.info(f"   Features: {self.scaler.n_features_in_}")
            else:
                logger.warning(f"⚠️ Scaler not found: {scaler_path}")
                logger.warning("   Predictions will use unscaled data")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_sensor_sequence(self, equipment_id: str, cycles: int = 30) -> np.ndarray:
      
        # FIXED: Add timestamp to seed for variation while keeping some consistency
        base_seed = hash(equipment_id) % 10000
        time_seed = int(time.time() * 1000) % 10000  
        combined_seed = (base_seed + time_seed) % 100000
        np.random.seed(combined_seed)
        
        sensor_data = np.zeros((cycles, self.n_features))
        
        # Simulate realistic sensor patterns with degradation
        # Values are in normalized range (-3 to 3, most between -1 and 1)
        for i in range(cycles):
            # Degradation factor increases over time
            degradation_factor = 1 - (i / cycles) * 0.3
            noise_level = 0.1 + (i / cycles) * 0.05  # Noise increases with degradation
            
            # Generate 13 sensor values matching C-MAPSS dataset patterns
            # Sensors based on turbofan engine measurements
            sensor_data[i] = np.array([
                # Sensor 12: Fan speed (sinusoidal with noise)
                (np.sin(i * 0.3) * 0.5 + np.random.normal(0, noise_level)) * degradation_factor,
                
                # Sensor 7: Total temperature (cosine pattern)
                (np.cos(i * 0.25) * 0.6 + np.random.normal(0, noise_level)) * degradation_factor,
                
                # Sensor 21: Bleed enthalpy (negative offset with sine)
                (np.sin(i * 0.2) * 0.4 + np.random.normal(0, noise_level) - 0.3) * degradation_factor,
                
                # Sensor 20: Pressure (cosine with offset)
                (np.cos(i * 0.35) * 0.3 + np.random.normal(0, noise_level)) * degradation_factor,
                
                # Sensor 14: Core speed (sine with positive offset)
                (np.sin(i * 0.15) * 0.6 + np.random.normal(0, noise_level) + 0.2) * degradation_factor,
                
                # Sensor 9: Static pressure
                (np.sin(i * 0.4) * 0.4 + np.random.normal(0, noise_level * 0.5)) * degradation_factor,
                
                # Sensor 13: Corrected fan speed
                (np.cos(i * 0.3) * 0.5 + np.random.normal(0, noise_level * 0.8)) * degradation_factor,
                
                # Sensor 8: Fuel flow
                (np.sin(i * 0.25) * 0.3 + np.random.normal(0, noise_level * 0.6)) * degradation_factor,
                
                # Sensor 3: LPC outlet temperature
                (np.cos(i * 0.2) * 0.4 + np.random.normal(0, noise_level * 0.7)) * degradation_factor,
                
                # Sensor 17: HPT coolant bleed
                (np.sin(i * 0.35) * 0.5 + np.random.uniform(-noise_level, noise_level)) * degradation_factor,
                
                # Operating Setting 1
                (np.cos(i * 0.4) * 0.3 + np.random.normal(0, noise_level * 0.5)) * degradation_factor,
                
                # Operating Setting 2
                (np.sin(i * 0.3) * 0.6 + np.random.normal(0, noise_level)) * degradation_factor,
                
                # Operating Setting 3
                (np.cos(i * 0.25) * 0.4 + np.random.normal(0, noise_level * 0.8)) * degradation_factor,
            ][:self.n_features])
        
        return sensor_data
    
    def predict_rul(self, sensor_sequence: np.ndarray) -> Dict:
        """
        Generate RUL prediction from 30-cycle sensor sequence
        
        Args:
            sensor_sequence: numpy array of shape (30, 13)
        
        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, loading now...")
            if not self.load_model():
                return self._get_fallback_prediction()
        
        try:
            # Validate input shape
            if sensor_sequence.shape != (self.sequence_length, self.n_features):
                raise ValueError(
                    f"Invalid sensor sequence shape. "
                    f"Expected ({self.sequence_length}, {self.n_features}), "
                    f"got {sensor_sequence.shape}"
                )
            
            # Store raw sensor data for health calculations
            raw_sensor_data = sensor_sequence.copy()
            
            # Reshape for LSTM: (batch_size=1, timesteps=30, features=13)
            sensor_sequence = np.expand_dims(sensor_sequence, axis=0)
            
            # Normalize using training scaler
            if self.scaler is not None:
                original_shape = sensor_sequence.shape
                reshaped = sensor_sequence.reshape(-1, sensor_sequence.shape[-1])
                scaled = self.scaler.transform(reshaped)
                sensor_sequence = scaled.reshape(original_shape)
            
            # Run LSTM prediction
            prediction = self.model.predict(sensor_sequence, verbose=0)
            
            # Extract RUL value and bound to realistic range
            rul_raw = float(prediction[0][0])
            rul = max(0, min(125, rul_raw))  # C-MAPSS dataset range: 0-125 cycles
            
            # Calculate confidence based on model certainty
            # Higher RUL = higher confidence (less degraded = more predictable)
            confidence = min(95, 60 + (rul / 125) * 35)
            
            # Determine equipment status
            status = self._determine_status(rul)
            
            # Calculate component health metrics using RAW (unscaled) sensor data
            health_metrics = self._calculate_health_metrics(raw_sensor_data, rul)
            
            logger.info(f"✅ Prediction: RUL={rul:.1f}, Status={status}, Confidence={confidence:.1f}%")
            logger.info(f"   Health: Speed={health_metrics['speedHealth']:.1f}%, "
                       f"Temp={health_metrics['temperatureHealth']:.1f}%, "
                       f"Pressure={health_metrics['pressureHealth']:.1f}%")
            
            return {
                'rul': round(rul, 1),
                'confidence': round(confidence, 1),
                'status': status,
                'health_metrics': health_metrics,
                'model_used': 'LSTM',
                'sensor_sequence': sensor_sequence[0].tolist()  # For debugging
            }
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_fallback_prediction()
    
    def _determine_status(self, rul: float) -> str:
        """Determine equipment status based on RUL"""
        if rul >= 70:
            return 'healthy'
        elif rul >= 50:
            return 'caution'
        else:
            return 'critical'
    
    def _calculate_health_metrics(self, sensor_data: np.ndarray, rul: float) -> Dict[str, float]:
    
        # Use last cycle readings for current health assessment
        last_cycle = sensor_data[-1]
        
        # Also calculate trend from first to last cycle
        first_cycle = sensor_data[0]
        
        # Extract key sensor readings (indices based on C-MAPSS dataset)
        # These are NORMALIZED values (typically between -1 and 1)
        fan_speed_current = last_cycle[0]       # Sensor 12
        temperature_current = last_cycle[1]     # Sensor 7
        pressure_current = last_cycle[3]        # Sensor 20
        
        fan_speed_initial = first_cycle[0]
        temperature_initial = first_cycle[1]
        pressure_initial = first_cycle[3]
        
        
        # Speed health
        # - Higher RUL = better health
        # - Less degradation from start to end = better health
        speed_degradation = abs(fan_speed_current - fan_speed_initial)
        speed_health = (rul / 125) * 50  # 0-50 points from RUL
        speed_health += (1 - speed_degradation) * 30  # 0-30 points from low degradation
        speed_health += (fan_speed_current + 1) * 10  # 0-20 points from normalized value
        speed_health = min(100, max(0, speed_health))
        
        # Temperature health (INVERSE - higher temperature = lower health)
        # In normalized data, higher values indicate higher temperatures
        temp_degradation = abs(temperature_current - temperature_initial)
        temp_health = (rul / 125) * 60  # 0-60 points from RUL
        temp_health += (1 - temp_degradation) * 20  # 0-20 points from stable temps
        temp_health -= abs(temperature_current) * 15  # Penalty for high temps
        temp_health = min(100, max(0, temp_health))
        
        # Pressure health
        pressure_degradation = abs(pressure_current - pressure_initial)
        pressure_health = (rul / 125) * 50  # 0-50 points from RUL
        pressure_health += (1 - pressure_degradation) * 30  # 0-30 points from stability
        pressure_health += (pressure_current + 1) * 10  # 0-20 points from normalized value
        pressure_health = min(100, max(0, pressure_health))
        
        return {
            'speedHealth': round(speed_health, 1),
            'temperatureHealth': round(temp_health, 1),
            'pressureHealth': round(pressure_health, 1),
        }
    
    def _get_fallback_prediction(self) -> Dict:
        """Return fallback prediction if model fails"""
        logger.warning("⚠️ Using fallback prediction (model unavailable)")
        return {
            'rul': 50.0,
            'confidence': 65.0,
            'status': 'caution',
            'health_metrics': {
                'speedHealth': 72.0,
                'temperatureHealth': 55.0,
                'pressureHealth': 68.0,
            },
            'model_used': 'Fallback',
        }
    
    def predict_from_equipment(self, equipment_id: str, equipment_type: str = 'Pump') -> Dict:
        """
        Complete prediction pipeline for equipment
        
        Args:
            equipment_id: Equipment identifier
            equipment_type: Type of equipment
            
        Returns:
            Complete prediction result with all metadata
        """
        # Generate 30-cycle sensor sequence (NOW WITH VARIATION)
        sensor_sequence = self.generate_sensor_sequence(equipment_id, cycles=30)
        
        # Make RUL prediction
        prediction = self.predict_rul(sensor_sequence)
        
        # Add metadata
        prediction['equipment_id'] = equipment_id
        prediction['equipment_type'] = equipment_type
        
        # Format sensor data for frontend
        prediction['sensor_data'] = self._format_sensor_data(sensor_sequence)
        
        return prediction
    
    def _format_sensor_data(self, sensor_sequence: np.ndarray) -> Dict:
        """Format sensor sequence for frontend visualization"""
        sensors = []
        
        for cycle_idx in range(len(sensor_sequence)):
            cycle_data = {
                'cycle': cycle_idx,
                'sensor_12': float(sensor_sequence[cycle_idx][0]),
                'sensor_7': float(sensor_sequence[cycle_idx][1]),
                'sensor_21': float(sensor_sequence[cycle_idx][2]),
                'sensor_20': float(sensor_sequence[cycle_idx][3]),
                'sensor_14': float(sensor_sequence[cycle_idx][4]),
            }
            sensors.append(cycle_data)
        
        return {'sensors': sensors}


# Global singleton instance
prediction_service = PredictionService()
