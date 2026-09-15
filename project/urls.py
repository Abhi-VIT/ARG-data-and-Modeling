from django.contrib import admin
from django.urls import include, path
from workspace import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/register/', views.register, name='register'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('api/', include('workspace.urls')),
    path('workspace-status/', views.workspace_status, name='workspace-status'),
    path('', views.index, name='workspace'),
]
