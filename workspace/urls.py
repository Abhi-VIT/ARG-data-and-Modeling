from django.urls import path
from . import api
from .deep import api as deep_api

urlpatterns = [
    path('deep/images/', deep_api.ImageUpload.as_view()),
    path('deep/train/', deep_api.Train.as_view()),
    path('deep/<uuid:job_id>/checkpoint/', deep_api.Artifact.as_view()),
    path('deep/<uuid:job_id>/output/', deep_api.Artifact.as_view(), {'output':True}),
    path('state/', api.State.as_view()), path('ingest/', api.Ingest.as_view()),
    path('preview/', api.Preview.as_view()), path('recipe/', api.Recipe.as_view()),
    path('jobs/', api.Jobs.as_view()), path('jobs/<uuid:job_id>/', api.JobDetail.as_view()),
    path('jobs/<uuid:job_id>/download/', api.Download.as_view()),
    path('jobs/<uuid:job_id>/report/', api.AnalysisReport.as_view()),
    path('models/catalog/', api.ModelCatalog.as_view()),
    path('models/<uuid:job_id>/download/', api.ModelArtifact.as_view()),
    path('models/<uuid:job_id>/output/', api.ModelArtifact.as_view(), {'output':True}),
    path('models/<uuid:job_id>/predict/', api.ModelPredict.as_view()),
]
