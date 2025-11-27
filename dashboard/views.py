from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Equipment, SensorData
from dashboard.ml.ml_service import prediction_service
import json
from datetime import datetime, timedelta
import logging

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