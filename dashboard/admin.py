"""
Django Admin Configuration for Dashboard
"""
from django.contrib import admin
from .models import (
    Equipment, SensorData, Prediction, 
    MaintenanceLog, Alert, SystemConfiguration
)


@admin.register(Equipment)
class EquipmentAdmin(admin.ModelAdmin):
    list_display = ['equipment_id', 'name', 'equipment_type', 'location', 'status', 'last_maintenance']
    list_filter = ['equipment_type', 'status', 'location']
    search_fields = ['equipment_id', 'name', 'location']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(SensorData)
class SensorDataAdmin(admin.ModelAdmin):
    list_display = ['equipment', 'timestamp', 'temperature', 'vibration_rms', 'current']
    list_filter = ['equipment', 'timestamp']
    search_fields = ['equipment__equipment_id']
    date_hierarchy = 'timestamp'
    readonly_fields = ['vibration_rms', 'vibration_peak']


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['equipment', 'timestamp', 'health_status', 'failure_probability', 'health_score', 'rul_days']
    list_filter = ['health_status', 'timestamp']
    search_fields = ['equipment__equipment_id']
    date_hierarchy = 'timestamp'
    readonly_fields = ['created_at']


@admin.register(MaintenanceLog)
class MaintenanceLogAdmin(admin.ModelAdmin):
    list_display = ['equipment', 'maintenance_type', 'status', 'scheduled_date', 'technician']
    list_filter = ['maintenance_type', 'status', 'scheduled_date']
    search_fields = ['equipment__equipment_id', 'description']
    date_hierarchy = 'scheduled_date'
    readonly_fields = ['created_at', 'updated_at']


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ['equipment', 'severity', 'title', 'is_acknowledged', 'created_at']
    list_filter = ['severity', 'is_acknowledged', 'created_at']
    search_fields = ['equipment__equipment_id', 'title', 'message']
    date_hierarchy = 'created_at'
    readonly_fields = ['created_at']


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(admin.ModelAdmin):
    list_display = ['config_key', 'description', 'updated_at', 'updated_by']
    search_fields = ['config_key', 'description']
    readonly_fields = ['updated_at']
