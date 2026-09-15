import uuid
from django.conf import settings
from django.db import models


class Workspace(models.Model):
    owner = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    active_dataset = models.ForeignKey('Dataset', null=True, blank=True, on_delete=models.SET_NULL, related_name='+')


class Upload(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    path = models.TextField()
    mime = models.CharField(max_length=120)
    created_at = models.DateTimeField(auto_now_add=True)


class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    cursor = models.PositiveIntegerField(default=0)
    original_schema = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)


class Revision(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='revisions')
    number = models.PositiveIntegerField()
    path = models.TextField()
    operation = models.JSONField(default=dict)
    profile = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['dataset', 'number'], name='unique_dataset_revision')]
        ordering = ['number']


class Job(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    kind = models.CharField(max_length=30)
    status = models.CharField(max_length=16, default='queued')
    progress = models.PositiveSmallIntegerField(default=0)
    message = models.CharField(max_length=500, default='Waiting for worker')
    payload = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    artifact = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['workspace'], condition=models.Q(status__in=['queued', 'running']),
                                               name='one_active_job_per_workspace')]
