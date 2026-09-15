from django.contrib import admin
from .models import Workspace, Dataset, Job

admin.site.register(Workspace)
admin.site.register(Dataset)
admin.site.register(Job)
