import json
from pathlib import Path
import zipfile
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from rest_framework.serializers import UUIDField
from rest_framework.views import APIView
from ..models import Job,Upload
from ..storage import private_path
from ..api import submit,user_workspace


class ImageUpload(APIView):
    throttle_scope='uploads'

    def post(self,request):
        from .images import safe_image_name
        workspace=user_workspace(request)
        if workspace.job_set.filter(status__in=['queued','running']).exists():raise ValidationError('Wait for the current job to finish.')
        archive=request.FILES.get('archive');files=request.FILES.getlist('images')
        if bool(archive)==bool(files):raise ValidationError('Choose one ZIP archive or a folder of images.')
        if getattr(request._request,'upload_too_large',False):raise ValidationError('Image upload exceeds the configured limit.')
        path=private_path(workspace.id,'.zip')
        try:
            if archive:
                if Path(archive.name).suffix.lower()!='.zip' or archive.size>settings.MAX_UPLOAD_BYTES or archive.content_type not in {'application/zip','application/x-zip-compressed','application/octet-stream'}:
                    raise ValidationError('Choose a ZIP archive within the upload size limit.')
                with path.open('wb') as destination:
                    for chunk in archive.chunks():destination.write(chunk)
                name=Path(archive.name).name[:255]
            else:
                paths=json.loads(request.data.get('paths','[]'))
                if not isinstance(paths,list) or len(paths)!=len(files) or not 12<=len(files)<=2000 or any(not isinstance(p,str) for p in paths):
                    raise ValidationError('Choose a class-organized folder with 12–2,000 images.')
                if sum(f.size for f in files)>settings.MAX_UPLOAD_BYTES:raise ValidationError('Combined images exceed the upload limit.')
                for p in paths:safe_image_name(p)
                if len(set(paths))!=len(paths):raise ValidationError('Image paths must be unique.')
                # Store multipart bytes without compression; image decoding is worker-only.
                with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as output:
                    for p,f in zip(paths,files):
                        with output.open(p,'w') as destination:
                            for chunk in f.chunks():destination.write(chunk)
                name='Image folder.zip'
            upload=Upload.objects.create(workspace=workspace,name=name,path=str(path),mime='application/zip')
        except Exception:
            path.unlink(missing_ok=True);raise
        return submit(workspace,'image_ingest',{'upload_id':str(upload.id)})


class Train(APIView):
    throttle_scope='jobs'

    def post(self,request):
        workspace=user_workspace(request);options=request.data.get('deep')
        if not isinstance(options,dict):raise ValidationError('Provide neural network settings.')
        payload={'deep':options}
        if options.get('architecture')=='cnn':
            image_id=UUIDField().run_validation(options.get('image_id'))
            get_object_or_404(Job,pk=image_id,workspace=workspace,kind='image_ingest',status='succeeded')
        else:
            dataset=workspace.active_dataset
            if not dataset:raise ValidationError('Import a tabular dataset or choose CNN and upload images.')
            revision=dataset.revisions.get(number=dataset.cursor)
            if request.data.get('dataset_id')!=str(dataset.id) or request.data.get('revision_id')!=revision.pk:
                raise ValidationError('The dataset changed. Refresh before training.')
            payload.update(dataset_id=str(dataset.id),revision=dataset.cursor,revision_id=revision.pk)
        return submit(workspace,'deep',payload)


class Artifact(APIView):
    def get(self,request,job_id,output=False):
        job=get_object_or_404(Job,pk=job_id,workspace__owner=request.user,kind='deep')
        if not job.artifact or (output and job.status!='succeeded'):raise ValidationError('This artifact is not ready yet.')
        path=Path(job.artifact).with_suffix('.csv' if output else '.pt')
        if not path.is_file():raise ValidationError('No completed checkpoint is available yet.')
        return FileResponse(open(path,'rb'),as_attachment=True,filename='deep-test-output.csv' if output else 'best-checkpoint.pt')
