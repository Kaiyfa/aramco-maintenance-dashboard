"""
Dashboard App URL Configuration
"""
from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Main dashboard route (homepage)
    path('', views.dashboard, name='dashboard'),
    
    # API endpoints
    path('api/index/', views.index, name='api_index'),
    path('api/equipment/', views.equipment_list, name='api_equipment_list'),
]