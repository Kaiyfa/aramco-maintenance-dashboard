from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Equipment, SensorData

# AUTHENTICATION VIEWS

def login_view(request):
    """
    Login page view
    """
    # If user is already logged in, redirect to dashboard
    if request.user.is_authenticated:
        return redirect('dashboard:dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            auth_login(request, user)
            
            # Redirect to 'next' parameter or dashboard
            next_url = request.POST.get('next') or request.GET.get('next') or 'dashboard:dashboard'
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'registration/login.html')


def logout_view(request):
    """
    Logout view
    """
    auth_logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')



# DASHBOARD VIEWS (Protected)


@login_required(login_url='login')
def dashboard(request):
    """
    Main dashboard view - REQUIRES LOGIN
    """
    context = {
        'equipment_count': Equipment.objects.count(),
        'sensor_data_count': SensorData.objects.count(),
        'user': request.user,
    }
    return render(request, 'dashboard/dashboard.html', context)



# API ENDPOINTS


@login_required(login_url='login')
def index(request):
    """
    API endpoint - returns JSON data
    """
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
    """
    Equipment list API endpoint
    """
    equipment = Equipment.objects.all().values('equipment_id', 'name', 'status')
    return JsonResponse(list(equipment), safe=False)