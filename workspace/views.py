from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods


@login_required
def index(request):
    return render(request, 'workspace/index.html')


@login_required
def workspace_status(request):
    from .models import Workspace
    workspace = Workspace.objects.filter(owner=request.user).first()
    active = workspace.job_set.filter(status__in=['queued', 'running']).first() if workspace else None
    return render(request, 'workspace/status.html', {'active_job': active})


@require_http_methods(['GET', 'POST'])
def register(request):
    form = UserCreationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        login(request, form.save())
        return redirect('workspace')
    return render(request, 'registration/register.html', {'form': form})
