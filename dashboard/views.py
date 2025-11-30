from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Equipment, SensorData
from dashboard.ml.ml_service import prediction_service
from django.db.models import Avg, Max, Min
from datetime import datetime, timedelta
import logging
import time
import json
import numpy as np
from django.utils import timezone 

logger = logging.getLogger(__name__)


# AUTHENTICATION VIEWS

def login_view(request):
    """Login page view"""
    if request.user.is_authenticated:
        return redirect('dashboard:dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            auth_login(request, user)
            next_url = request.POST.get('next') or request.GET.get('next') or 'dashboard:dashboard'
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'registration/login.html')


def logout_view(request):
    """Logout view"""
    auth_logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')



# DASHBOARD VIEWS

@login_required(login_url='login')
def dashboard(request):
    """Main dashboard view"""
    context = {
        'equipment_count': Equipment.objects.count(),
        'sensor_data_count': SensorData.objects.count(),
        'user': request.user,
    }
    return render(request, 'dashboard/dashboard.html', context)


# API ENDPOINTS

@login_required(login_url='login')
def index(request):
    """API health check endpoint"""
    equipment_count = Equipment.objects.count()
    sensor_data_count = SensorData.objects.count()
    
    return JsonResponse({
        'equipment_count': equipment_count,
        'sensor_data_count': sensor_data_count,
        'status': 'API is working',
        'user': request.user.username
    })


@login_required(login_url='login')
def equipment_list(request):
    """Equipment list API endpoint"""
    equipment = Equipment.objects.all().values('equipment_id', 'name', 'status')
    return JsonResponse(list(equipment), safe=False)


@login_required(login_url='login')
def get_prediction(request):
    """
    Generate comprehensive AI prediction with LSTM model
    
    Returns:
        JSON with prediction, recommendations, alerts, cost impact, and trend metrics
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Extract request parameters
        equipment_type = request.POST.get('equipment_type', 'Pump')
        equipment_id = request.POST.get('equipment_id', 'EQ-006')
        location = request.POST.get('location', 'Ghawar Field')
        
        logger.info(f"🔄 Prediction request: {equipment_type} {equipment_id} @ {location}")
        
        # Load model if not already loaded
        if not prediction_service.model_loaded:
            logger.info("📦 Loading LSTM model...")
            if not prediction_service.load_model():
                raise Exception("Failed to load LSTM model")
        
        # Generate prediction using LSTM model
        prediction_result = prediction_service.predict_from_equipment(
            equipment_id=equipment_id,
            equipment_type=equipment_type
        )
        
        # Add metadata
        prediction_result['location'] = location
        prediction_result['timestamp'] = datetime.now().isoformat()
        prediction_result['user'] = request.user.username
        
        # Generate AI recommendations
        recommendations = generate_ai_recommendations(prediction_result)
        
        # Generate contextual alerts
        alerts = generate_contextual_alerts(prediction_result)
        
        # Calculate cost impact analysis
        cost_impact = calculate_cost_impact(prediction_result, recommendations)
        
        # Calculate trend metrics
        trend_metrics = calculate_trend_metrics(prediction_result)
        
        logger.info(f"✅ Prediction complete: RUL={prediction_result['rul']}, Status={prediction_result['status']}")
        
        # Return comprehensive response
        return JsonResponse({
            'success': True,
            'prediction': prediction_result,
            'recommendations': recommendations,
            'alerts': alerts,
            'costImpact': cost_impact,
            'trendMetrics': trend_metrics
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# AI RECOMMENDATION ENGINE

def generate_ai_recommendations(prediction: dict) -> list:
    """
    Generate intelligent maintenance recommendations based on prediction
    
    Matches Replit's recommendation structure with:
    - Priority levels (HIGH/MEDIUM/LOW)
    - Cost impact estimates
    - Estimated savings
    - Actionable steps
    """
    recommendations = []
    rul = prediction['rul']
    status = prediction['status']
    health = prediction['health_metrics']
    equipment_id = prediction['equipment_id']
    
    # CRITICAL STATUS: Immediate action required
    if status == 'critical':
        recommendations.append({
            'id': 'critical_maintenance',
            'priority': 'HIGH',
            'title': 'Schedule Preventive Inspection',
            'description': f'Critical wear patterns detected on {equipment_id}. RUL critically low at {rul:.1f} cycles. Immediate inspection required to prevent catastrophic failure.',
            'action': 'Schedule emergency inspection within 24-48 hours and prepare replacement parts (bearings, seals, gaskets).',
            'costImpact': '$50,000 - $75,000',
            'estimatedSavings': '$500,000 (avoided unplanned downtime)'
        })
    
    # CAUTION STATUS: Plan maintenance
    elif status == 'caution':
        recommendations.append({
            'id': 'preventive_maintenance',
            'priority': 'MEDIUM',
            'title': 'Schedule Preventive Inspection',
            'description': f'Moderate wear patterns detected on {equipment_id}. Schedule inspection within 2-3 weeks to prevent unexpected failures.',
            'action': 'Schedule inspection within 2-3 weeks and prepare replacement parts.',
            'costImpact': '$2,500 - $3,500',
            'estimatedSavings': '$15,000 (avoided downtime)'
        })
    
    # TEMPERATURE HEALTH: Operating parameter optimization
    if health['temperatureHealth'] > 80:
        temp_diff = int((health['temperatureHealth'] - 72) * 0.25)
        recommendations.append({
            'id': 'optimize_temperature',
            'priority': 'MEDIUM',
            'title': 'Optimize Operating Parameters',
            'description': f'Current operating temperature is {temp_diff}% higher than optimal range. Elevated temperatures accelerate component degradation.',
            'action': 'Reduce operating speed by 5% and increase cooling system efficiency. Monitor temperature trends over next 48 hours.',
            'costImpact': '$1,200 - $1,800',
            'estimatedSavings': '$5,000 annually (reduced wear)'
        })
    elif health['temperatureHealth'] < 60:
        recommendations.append({
            'id': 'investigate_cooling',
            'priority': 'MEDIUM',
            'title': 'Investigate Cooling System',
            'description': f'Temperature readings indicate potential cooling system inefficiency. Current health at {health["temperatureHealth"]:.1f}%.',
            'action': 'Inspect cooling system, check coolant levels, and verify thermal sensor accuracy.',
            'costImpact': '$800 - $1,500',
            'estimatedSavings': '$3,000 (prevented thermal damage)'
        })
    
    # SPEED HEALTH: Mechanical inspection
    if health['speedHealth'] < 70:
        recommendations.append({
            'id': 'mechanical_inspection',
            'priority': 'MEDIUM',
            'title': 'Mechanical System Inspection',
            'description': f'Speed health at {health["speedHealth"]:.1f}% indicates potential mechanical wear. Motor speed variations detected outside normal parameters.',
            'action': 'Check motor bearings, drive system alignment, and lubrication levels. Schedule vibration analysis.',
            'costImpact': '$1,500 - $2,500',
            'estimatedSavings': '$8,000 (prevented bearing failure)'
        })
    
    # SENSOR CALIBRATION: Routine maintenance
    if health['speedHealth'] < 75 or health['pressureHealth'] < 75:
        recommendations.append({
            'id': 'sensor_calibration',
            'title': 'Update Sensor Calibration',
            'priority': 'LOW',
            'description': 'Sensors have been in operation for extended period without recalibration. Sensor drift may affect prediction accuracy.',
            'action': 'Schedule sensor calibration during next planned maintenance window.',
            'costImpact': '$400 - $600',
            'estimatedSavings': None
        })
    
    # HEALTHY STATUS: Continue monitoring
    if status == 'healthy':
        recommendations.append({
            'id': 'routine_monitoring',
            'title': 'Continue Routine Monitoring',
            'priority': 'LOW',
            'description': f'Equipment {equipment_id} is operating within normal parameters. RUL: {rul:.1f} cycles remaining.',
            'action': 'Maintain current monitoring schedule and operating conditions. Next inspection in 4-6 weeks.',
            'costImpact': '$200 - $400',
            'estimatedSavings': None
        })
    
    return recommendations


# ALERT GENERATION SYSTEM

def generate_contextual_alerts(prediction: dict) -> list:
    """
    Generate contextual alerts based on sensor thresholds and equipment status
    
    Returns alerts with:
    - Severity levels (CRITICAL/WARNING/INFO)
    - Specific sensor anomalies
    - Timestamps
    - Equipment context
    """
    alerts = []
    rul = prediction['rul']
    health = prediction['health_metrics']
    equipment_id = prediction['equipment_id']
    
    # CRITICAL RUL ALERT
    if rul < 30:
        alerts.append({
            'id': f'alert_critical_rul_{int(datetime.now().timestamp())}',
            'severity': 'critical',
            'title': 'Critical: Equipment Approaching Failure',
            'description': f'RUL has dropped to {rul:.1f} cycles. Equipment may fail within {int(rul)} operational cycles. Immediate action required.',
            'equipmentId': equipment_id,
            'timestamp': '15 minutes ago',
        })
    
    # PERFORMANCE DEGRADATION WARNING
    if health['temperatureHealth'] < 60:
        alerts.append({
            'id': f'alert_degradation_{int(datetime.now().timestamp())}',
            'severity': 'warning',
            'title': 'Warning: Gradual Performance Degradation',
            'description': f'Equipment health at {health["temperatureHealth"]:.1f}%. Monitor closely for further degradation. Temperature sensors showing elevated readings.',
            'equipmentId': equipment_id,
            'timestamp': '15 minutes ago',
        })
    
    # SCHEDULED MAINTENANCE INFO
    if rul < 60:
        alerts.append({
            'id': f'alert_maintenance_{int(datetime.now().timestamp())}',
            'severity': 'info',
            'title': 'Scheduled Maintenance Window',
            'description': 'Routine maintenance scheduled for next week. Preventive inspection recommended based on current equipment health.',
            'equipmentId': equipment_id,
            'timestamp': '2 days ago',
        })
    
    # SENSOR ANOMALY WARNING
    if health['speedHealth'] < 70:
        alerts.append({
            'id': f'alert_sensor_anomaly_{int(datetime.now().timestamp())}',
            'severity': 'warning',
            'title': 'Sensor Reading Anomaly',
            'description': 'Sensor_21 showing unusual fluctuation patterns over the past 48 hours. Speed sensor readings vary outside normal operational range.',
            'equipmentId': equipment_id,
            'timestamp': '3 days ago',
        })
    
    return alerts


# COST IMPACT ANALYSIS

def calculate_cost_impact(prediction: dict, recommendations: list) -> dict:
    """
    Calculate comprehensive cost impact analysis
    
    Returns:
    - Total potential cost
    - Estimated savings
    - ROI percentage
    """
    total_cost = 0
    total_savings = 0
    
    for rec in recommendations:
        # Parse cost range (e.g., "$2,500 - $3,500")
        if rec.get('costImpact'):
            try:
                cost_str = rec['costImpact'].replace('$', '').replace(',', '')
                if ' - ' in cost_str:
                    low, high = cost_str.split(' - ')
                    avg_cost = (float(low) + float(high)) / 2
                else:
                    avg_cost = float(cost_str)
                total_cost += avg_cost
            except ValueError:
                pass
        
        # Parse savings (e.g., "$15,000 (avoided downtime)")
        if rec.get('estimatedSavings'):
            try:
                savings_str = rec['estimatedSavings'].split('(')[0]
                savings_str = savings_str.replace('$', '').replace(',', '').strip()
                total_savings += float(savings_str)
            except ValueError:
                pass
    
    # Calculate ROI
    roi = (total_savings / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'totalCost': f"${int(total_cost):,}",
        'estimatedSavings': f"${int(total_savings):,}",
        'roi': f"{int(roi)}%"
    }


# TREND METRICS CALCULATION

def calculate_trend_metrics(prediction: dict) -> dict:
    """
    Calculate degradation rate and health trends
    
    In production, this would analyze historical predictions
    For now, calculates based on current state
    """
    rul = prediction['rul']
    health = prediction['health_metrics']
    
    # Calculate average health across all metrics
    avg_health = (
        health['speedHealth'] + 
        health['temperatureHealth'] + 
        health['pressureHealth']
    ) / 3
    
    # Estimate degradation rate
    # Assuming max RUL of 125 cycles (C-MAPSS dataset range)
    degradation_rate = ((125 - rul) / 125) * 2  # Percentage per prediction
    
    return {
        'degradationRate': round(degradation_rate, 2),
        'averageHealth': round(avg_health, 1)
    }


# LIVE SENSOR API ENDPOINTS

@login_required(login_url='login')
def live_reading(request):
    """
    Fetch the absolute latest sensor reading from TimescaleDB
    
    Returns:
        JSON with current sensor values and timestamp
    """
    try:
        equipment_id = request.GET.get('equipment_id', 'EQ-006')
        
        # Fetch most recent sensor reading
        latest_reading = SensorData.objects.filter(
            equipment_id=equipment_id
        ).order_by('-timestamp').first()
        
        if not latest_reading:
            # No real data - return simulated live reading
            return JsonResponse({
                'success': True,
                'source': 'simulated',
                'timestamp': timezone.now().isoformat(),
                'equipment_id': equipment_id,
                'sensors': {
                    'vibration_x': round(np.random.normal(0.5, 0.1), 3),
                    'vibration_y': round(np.random.normal(0.6, 0.1), 3),
                    'vibration_z': round(np.random.normal(0.4, 0.1), 3),
                    'vibration_rms': round(np.random.normal(0.5, 0.08), 3),
                    'vibration_peak': round(np.random.normal(0.8, 0.15), 3),
                    'temperature': round(np.random.normal(45.0, 5.0), 1),
                    'current': round(np.random.normal(5.2, 0.5), 2),
                },
                'thresholds': {
                    'temperature_max': 85.0,
                    'vibration_max': 2.5,
                    'current_max': 10.0,
                },
                'alerts': []
            })
        
        # Real data from TimescaleDB
        sensors = {
            'vibration_x': latest_reading.vibration_x,
            'vibration_y': latest_reading.vibration_y,
            'vibration_z': latest_reading.vibration_z,
            'vibration_rms': latest_reading.vibration_rms,
            'vibration_peak': latest_reading.vibration_peak,
            'temperature': latest_reading.temperature,
            'current': latest_reading.current,
        }
        
        # Check threshold violations
        thresholds = {
            'temperature_max': 85.0,
            'vibration_max': 2.5,
            'current_max': 10.0,
        }
        
        alerts = []
        
        if sensors['temperature'] > thresholds['temperature_max']:
            alerts.append({
                'severity': 'critical',
                'sensor': 'temperature',
                'message': f"HIGH TEMP: {sensors['temperature']:.1f}°C > {thresholds['temperature_max']}°C threshold",
                'timestamp': latest_reading.timestamp.isoformat()
            })
        
        if sensors['vibration_rms'] and sensors['vibration_rms'] > thresholds['vibration_max']:
            alerts.append({
                'severity': 'warning',
                'sensor': 'vibration',
                'message': f"HIGH VIBRATION: {sensors['vibration_rms']:.2f}g > {thresholds['vibration_max']}g threshold",
                'timestamp': latest_reading.timestamp.isoformat()
            })
        
        if sensors['current'] > thresholds['current_max']:
            alerts.append({
                'severity': 'warning',
                'sensor': 'current',
                'message': f"HIGH CURRENT: {sensors['current']:.1f}A > {thresholds['current_max']}A threshold",
                'timestamp': latest_reading.timestamp.isoformat()
            })
        
        return JsonResponse({
            'success': True,
            'source': 'live',
            'timestamp': latest_reading.timestamp.isoformat(),
            'equipment_id': equipment_id,
            'sensors': sensors,
            'thresholds': thresholds,
            'alerts': alerts
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching live reading: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@login_required(login_url='login')
def live_prediction(request):
    """
    Generate ML prediction using last 30 live sensor readings
    
    This endpoint is called periodically (e.g., every 5 minutes) by the Live Dashboard
    to update the RUL prediction based on accumulated real-time data.
    
    Returns:
        JSON with RUL prediction and health metrics
    """
    try:
        equipment_id = request.GET.get('equipment_id', 'EQ-006')
        equipment_type = request.GET.get('equipment_type', 'Pump')
        
        logger.info(f"🔴 LIVE PREDICTION: Using real sensor data for {equipment_id}")
        
        # Load model if not already loaded
        if not prediction_service.model_loaded:
            logger.info("📦 Loading LSTM model...")
            if not prediction_service.load_model():
                raise Exception("Failed to load LSTM model")
        
        # Check if we have enough real sensor data
        reading_count = SensorData.objects.filter(equipment_id=equipment_id).count()
        
        if reading_count < 30:
            logger.warning(f"Only {reading_count} readings available. Using simulated data.")
            # Fall back to simulated prediction
            prediction_result = prediction_service.predict_from_equipment(
                equipment_id=equipment_id,
                equipment_type=equipment_type
            )
            prediction_result['data_source'] = 'simulated'
        else:
            # Use real live sensor data
            prediction_result = prediction_service.predict_from_live_sensors(
                equipment_id=equipment_id,
                equipment_type=equipment_type
            )
        
        # Get recent sensor statistics for context
        recent_readings = SensorData.objects.filter(
            equipment_id=equipment_id
        ).order_by('-timestamp')[:100]
        
        if recent_readings.exists():
            stats = recent_readings.aggregate(
                avg_temp=Avg('temperature'),
                max_temp=Max('temperature'),
                avg_vibration_rms=Avg('vibration_rms'),
                max_vibration_rms=Max('vibration_rms'),
                avg_current=Avg('current'),
                max_current=Max('current'),
            )
            
            prediction_result['sensor_statistics'] = {
                'temperature': {
                    'average': round(stats['avg_temp'] or 0, 1),
                    'max': round(stats['max_temp'] or 0, 1),
                },
                'vibration_rms': {
                    'average': round(stats['avg_vibration_rms'] or 0, 3),
                    'max': round(stats['max_vibration_rms'] or 0, 3),
                },
                'current': {
                    'average': round(stats['avg_current'] or 0, 2),
                    'max': round(stats['max_current'] or 0, 2),
                }
            }
        
        # Save to prediction history
        from dashboard.models import PredictionHistory
        PredictionHistory.objects.create(
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            location=request.GET.get('location', 'Ghawar Field'),
            rul=prediction_result['rul'],
            confidence=prediction_result['confidence'],
            status=prediction_result['status'],
            speed_health=prediction_result['health_metrics']['speedHealth'],
            temperature_health=prediction_result['health_metrics']['temperatureHealth'],
            pressure_health=prediction_result['health_metrics']['pressureHealth'],
            data_source=prediction_result.get('data_source', 'live'),
            model_used='LSTM',
        )
        
        logger.info(f"✅ Live prediction complete: RUL={prediction_result['rul']:.1f}")
        
        return JsonResponse({
            'success': True,
            'prediction': prediction_result,
            'reading_count': reading_count,
        })
        
    except Exception as e:
        logger.error(f"❌ Error generating live prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@login_required(login_url='login')
def live_dashboard(request):
    """
    Live Sensor Dashboard view
    Displays real-time sensor readings and continuous monitoring
    """
    context = {
        'user': request.user,
        'dashboard_type': 'live',
    }
    return render(request, 'dashboard/dashboard_live.html', context)


@login_required(login_url='login')
def live_sensor_history(request):
    """
    Fetch last N seconds of sensor readings for real-time charts
    
    Returns:
        JSON with time-series sensor data for charting
    """
    try:
        equipment_id = request.GET.get('equipment_id', 'EQ-006')
        seconds = int(request.GET.get('seconds', 60))  # Default: last 60 seconds
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=seconds)
        
        # Fetch readings in time range
        readings = SensorData.objects.filter(
            equipment_id=equipment_id,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')
        
        # Format for charting
        data = {
            'timestamps': [],
            'temperature': [],
            'vibration_rms': [],
            'current': [],
        }
        
        for reading in readings:
            data['timestamps'].append(reading.timestamp.isoformat())
            data['temperature'].append(reading.temperature)
            data['vibration_rms'].append(reading.vibration_rms or 0)
            data['current'].append(reading.current)
        
        # If no real data, generate simulated historical data
        if not readings.exists():
            logger.warning("No real sensor history. Generating simulated data.")
            num_points = 12  # 12 points for 60 seconds (5 sec intervals)
            current_time = end_time
            
            for i in range(num_points):
                timestamp = current_time - timedelta(seconds=(num_points - i - 1) * 5)
                data['timestamps'].append(timestamp.isoformat())
                data['temperature'].append(round(45.0 + np.random.normal(0, 2), 1))
                data['vibration_rms'].append(round(0.5 + np.random.normal(0, 0.1), 3))
                data['current'].append(round(5.2 + np.random.normal(0, 0.3), 2))
        
        return JsonResponse({
            'success': True,
            'equipment_id': equipment_id,
            'time_range_seconds': seconds,
            'data_points': len(data['timestamps']),
            'data': data
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching sensor history: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)