"""
Dashboard Views
"""
from django.shortcuts import render
from django.http import JsonResponse
from .models import Equipment, SensorData

def index(request):
    equipment_count = Equipment.objects.count()
    sensor_data_count = SensorData.objects.count()
    
    return JsonResponse({
        'equipment_count': equipment_count,
        'sensor_data_count': sensor_data_count,
        'status': 'API is working'
    })

def equipment_list(request):
    equipment = Equipment.objects.all().values('equipment_id', 'name', 'status')
    return JsonResponse(list(equipment), safe=False)
