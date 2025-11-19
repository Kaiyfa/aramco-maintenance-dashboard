"""
Dashboard Views
"""
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

def index(request):
    """Landing page"""
    return render(request, 'dashboard/index.html')

@login_required
def dashboard_view(request):
    """Main dashboard view"""
    context = {
        'page_title': 'Predictive Maintenance Dashboard'
    }
    return render(request, 'dashboard/dashboard.html', context)
