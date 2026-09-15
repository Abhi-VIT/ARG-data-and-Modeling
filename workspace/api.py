import json
from pathlib import Path
import duckdb
import numpy as np
from django.conf import settings
from django.db import IntegrityError, transaction
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView, exception_handler
from .models import Job, Upload, Workspace
from .ingestion import DataError, MIMES, ROW_ID, validate_file
from .storage import private_path
from .tasks import run_job


def api_exception_handler(exc, context):
    if isinstance(exc, (DataError, ValueError, TypeError, KeyError)):
        return Response({'detail': str(exc)[:500]}, status=400)
    from redis.exceptions import RedisError
    if isinstance(exc, RedisError):
        return Response({'detail': 'Redis is unavailable. Start Redis and retry.'}, status=503)
    return exception_handler(exc, context)


def user_workspace(request):
    return Workspace.objects.get_or_create(owner=request.user)[0]


def serialize_job(job):
    return {'id': str(job.id), 'kind': job.kind, 'status': job.status, 'progress': job.progress,
            'message': job.message, 'result': job.result, 'created_at': job.created_at.isoformat()}


def submit(workspace, kind, payload):
    try:
        with transaction.atomic():
            job = Job.objects.create(workspace=workspace, kind=kind, payload=payload)
    except IntegrityError:
        raise ValidationError('A workspace job is already running. Wait for it to finish.')
    try:
        run_job.delay(str(job.id))
    except Exception:
        Job.objects.filter(pk=job.id, status='queued').update(
            status='failed', message='Cannot reach the job broker. Start Redis and the Celery worker, then retry.')
    job.refresh_from_db()
    return Response(serialize_job(job), status=202 if job.status != 'failed' else 503)


class State(APIView):
    def get(self, request):
        workspace = user_workspace(request)
        dataset = workspace.active_dataset
        data = None
        if dataset:
            revision = dataset.revisions.get(number=dataset.cursor)
            history = list(dataset.revisions.values('number', 'operation', 'created_at'))
            data = {'id': str(dataset.id), 'name': dataset.name, 'revision': dataset.cursor, 'revision_id': revision.pk,
                    'profile': revision.profile, 'history': history,
                    'can_undo': dataset.cursor > 0, 'can_redo': dataset.revisions.filter(number__gt=dataset.cursor).exists()}
        jobs = workspace.job_set.order_by('-created_at')[:10]
        return Response({'dataset': data, 'jobs': [serialize_job(job) for job in jobs],
                         'analyses': [serialize_job(job) for job in workspace.job_set.filter(kind='analysis', status='succeeded').order_by('-created_at')[:30]],
                         'models': [serialize_job(job) for job in workspace.job_set.filter(kind__in=['model','predict'], status='succeeded').order_by('-created_at')[:30]],
                         'max_upload_mb': settings.MAX_UPLOAD_BYTES // 1024**2,
                         'import_hosts': settings.IMPORT_URL_HOSTS})


class Ingest(APIView):
    throttle_scope = 'uploads'

    def post(self, request):
        workspace = user_workspace(request)
        if workspace.job_set.filter(status__in=['queued', 'running']).exists():
            raise ValidationError('Wait for the current workspace job to finish before uploading.')
        options = request.data.get('options', {})
        if isinstance(options, str):
            options = json.loads(options)
        if not isinstance(options, dict):
            raise ValidationError('Parsing options must be an object.')
        if request.data.get('url'):
            return submit(workspace, 'url', {'url': str(request.data['url'])[:2048], 'options': options})
        if request.data.get('upload_id'):
            upload = get_object_or_404(Upload, pk=request.data['upload_id'], workspace=workspace)
        else:
            uploaded = request.FILES.get('file')
            if getattr(request._request, 'upload_too_large', False):
                raise ValidationError('Upload exceeds the configured size limit.')
            if not uploaded:
                raise ValidationError('Choose a file to upload.')
            if uploaded.size > settings.MAX_UPLOAD_BYTES:
                raise ValidationError('Upload exceeds the configured size limit.')
            name = Path(uploaded.name).name[:255]
            ext = Path(name).suffix.lower()
            if ext not in MIMES:
                raise ValidationError('Unsupported file extension.')
            path = private_path(workspace.id, ext)
            try:
                with path.open('wb') as output:
                    for chunk in uploaded.chunks():
                        output.write(chunk)
                # Header checks are small; workbook expansion checks and parsing run in the worker.
                mime = uploaded.content_type or 'application/octet-stream'
                if mime.split(';')[0] not in MIMES[ext] | {'application/octet-stream', ''}:
                    raise DataError('The reported file type does not match its extension.')
                upload = Upload.objects.create(workspace=workspace, name=name, path=str(path), mime=mime)
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return submit(workspace, 'ingest', {'upload_id': str(upload.id), 'options': options})


class Jobs(APIView):
    throttle_scope = 'jobs'

    def post(self, request):
        workspace = user_workspace(request)
        dataset = workspace.active_dataset
        if not dataset:
            raise ValidationError('Import a dataset first.')
        kind = request.data.get('kind')
        if kind not in {'model', 'analysis', 'clean', 'detect', 'replay', 'export', 'undo', 'redo', 'clear'}:
            raise ValidationError('Unsupported job type.')
        if request.data.get('dataset_id') != str(dataset.id) or request.data.get('revision') != dataset.cursor:
            raise ValidationError('The dataset changed. Refresh before submitting another operation.')
        current_revision=dataset.revisions.get(number=dataset.cursor)
        if request.data.get('revision_id',current_revision.pk)!=current_revision.pk:
            raise ValidationError('This revision was replaced after Undo. Refresh before submitting.')
        payload = {key: request.data[key] for key in ['model', 'analysis', 'operation', 'recipe', 'format', 'detection_job'] if key in request.data}
        if kind == 'model' and (not isinstance(payload.get('model'),dict) or not payload['model'].get('features')):
            raise ValidationError('Choose a model and select feature columns first.')
        if kind == 'analysis' and (not isinstance(payload.get('analysis'), dict) or not payload['analysis'].get('columns')):
            raise ValidationError('Choose an analysis and select columns first.')
        if kind in {'clean', 'detect'} and not isinstance(payload.get('operation'), dict):
            raise ValidationError('Provide a cleaning operation.')
        payload.update(dataset_id=str(dataset.id), revision=dataset.cursor, revision_id=current_revision.pk)
        return submit(workspace, kind, payload)


class JobDetail(APIView):
    def get(self, request, job_id):
        job = get_object_or_404(Job, pk=job_id, workspace__owner=request.user)
        return Response(serialize_job(job))


class Download(APIView):
    def get(self, request, job_id):
        job = get_object_or_404(Job, pk=job_id, workspace__owner=request.user, status='succeeded', kind__in=['export', 'analysis', 'model', 'predict'])
        if not job.artifact or not Path(job.artifact).is_file():
            raise ValidationError('This export is no longer available. Create another export.')
        return FileResponse(open(job.artifact, 'rb'), as_attachment=True, filename=job.result['filename'])


class AnalysisReport(APIView):
    def get(self, request, job_id):
        job = get_object_or_404(Job, pk=job_id, workspace__owner=request.user, status='succeeded', kind__in=['analysis','model','predict'])
        if not job.artifact or not Path(job.artifact).is_file():
            raise ValidationError('Report file is no longer available. Run the analysis again.')
        return FileResponse(open(job.artifact, 'rb'), content_type='application/json')


class ModelCatalog(APIView):
    def get(self, request):
        from .ml.catalog import CATALOG
        return Response({'models':list(CATALOG.values())})


class ModelArtifact(APIView):
    def get(self, request, job_id, output=False):
        job=get_object_or_404(Job,pk=job_id,workspace__owner=request.user,status='succeeded',kind__in=['model','predict'] if output else ['model'])
        path=Path(job.artifact).with_suffix('.csv' if output else '.pkl')
        if not path.is_file():raise ValidationError('This model or output file is unavailable.')
        return FileResponse(open(path,'rb'),as_attachment=True,filename=('predictions' if job.kind=='predict' else 'model-output')+'.csv' if output else 'trained-model.pkl')


class ModelPredict(APIView):
    throttle_scope='jobs'

    def post(self, request, job_id):
        source=get_object_or_404(Job,pk=job_id,workspace__owner=request.user,kind='model',status='succeeded')
        if not source.result.get('can_predict'):raise ValidationError('This estimator cannot score new observations.')
        workspace=source.workspace
        if workspace.job_set.filter(status__in=['queued','running']).exists():raise ValidationError('Wait for the current workspace job to finish.')
        uploaded=request.FILES.get('file')
        if not uploaded or uploaded.size>settings.MAX_UPLOAD_BYTES:raise ValidationError('Choose a prediction data file within the upload size limit.')
        name=Path(uploaded.name).name[:255];ext=Path(name).suffix.lower()
        mime=(uploaded.content_type or 'application/octet-stream').split(';')[0]
        if ext not in MIMES or mime not in MIMES[ext]|{'application/octet-stream',''}:raise ValidationError('Unsupported prediction file type. Upload data, not a serialized model.')
        options=request.data.get('options',{})
        if isinstance(options,str):options=json.loads(options)
        if not isinstance(options,dict):raise ValidationError('Prediction parsing options must be an object.')
        threshold=float(request.data.get('threshold',.5))
        if not 0<=threshold<=1:raise ValidationError('Threshold must be between 0 and 1.')
        path=private_path(workspace.id,ext)
        with path.open('wb') as destination:
            for chunk in uploaded.chunks():destination.write(chunk)
        upload=Upload.objects.create(workspace=workspace,name=name,path=str(path),mime=mime)
        return submit(workspace,'predict',{'model_id':str(source.id),'upload_id':str(upload.id),'options':options,'threshold':threshold})


class Recipe(APIView):
    def get(self, request):
        workspace = user_workspace(request)
        dataset = workspace.active_dataset
        if not dataset:
            raise ValidationError('Import a dataset first.')
        operations = list(dataset.revisions.filter(number__gt=0, number__lte=dataset.cursor).values_list('operation', flat=True))
        response = Response({'version': 1, 'schema': dataset.original_schema, 'operations': operations})
        response['Content-Disposition'] = 'attachment; filename="cleaning-recipe.json"'
        return response


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


class Preview(APIView):
    def get(self, request):
        workspace = user_workspace(request)
        dataset = workspace.active_dataset
        if not dataset:
            raise ValidationError('Import a dataset first.')
        revision = dataset.revisions.get(number=dataset.cursor)
        columns = [c['name'] for c in revision.profile['columns']]
        page = max(1, int(request.query_params.get('page', 1)))
        size = max(10, min(100, int(request.query_params.get('page_size', 50))))
        search = request.query_params.get('q', '')[:200]
        filters = json.loads(request.query_params.get('filters', '{}'))
        if not isinstance(filters, dict) or not set(filters) <= set(columns):
            raise ValidationError('Unknown filter columns.')
        conditions, params = [], []
        if search:
            conditions.append('(' + ' OR '.join(f'contains(lower(CAST({quote_identifier(c)} AS VARCHAR)), lower(?))' for c in columns) + ')')
            params.extend([search] * len(columns))
        for column, value in filters.items():
            conditions.append(f'contains(lower(CAST({quote_identifier(column)} AS VARCHAR)), lower(?))')
            params.append(str(value)[:200])
        sort = request.query_params.get('sort', ROW_ID)
        if sort not in columns + [ROW_ID]:
            raise ValidationError('Unknown sort column.')
        direction = 'DESC' if request.query_params.get('direction') == 'desc' else 'ASC'
        where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
        with duckdb.connect(config={'threads': 2, 'memory_limit': '512MB', 'enable_external_access': True}) as db:
            db.read_parquet(revision.path).create_view('data')
            count = db.execute('SELECT count(*) FROM data' + where, params).fetchone()[0]
            page = min(page, max(1, (count + size - 1) // size))
            rows = db.execute(f'SELECT * FROM data{where} ORDER BY {quote_identifier(sort)} {direction} NULLS LAST, '
                              f'{quote_identifier(ROW_ID)} ASC LIMIT ? OFFSET ?', params + [size, (page - 1) * size]).fetchdf()
            flagged = set()
            detection_id = request.query_params.get('detection')
            if detection_id:
                detection = get_object_or_404(Job, pk=detection_id, workspace=workspace, kind='detect', status='succeeded')
                if detection.result.get('dataset_id') == str(dataset.id) and detection.result.get('revision') == dataset.cursor:
                    db.read_parquet(detection.artifact).create_view('flags')
                    ids = rows[ROW_ID].tolist()
                    if ids:
                        flagged = {r[0] for r in db.execute(f'SELECT {quote_identifier(ROW_ID)} FROM flags WHERE '
                                                          f'{quote_identifier(ROW_ID)} IN ({",".join("?" for _ in ids)})', ids).fetchall()}
        for column in columns:
            if column in rows.select_dtypes(include=['number']).columns:
                infinite = np.isinf(rows[column])
                if infinite.any():
                    values = rows[column].astype(object)
                    values.loc[infinite] = rows.loc[infinite, column].map(lambda value: 'Infinity' if value > 0 else '-Infinity')
                    rows[column] = values
        records = json.loads(rows.to_json(orient='records', date_format='iso', double_precision=15))
        output = []
        for record in records:
            row_id = record.pop(ROW_ID)
            # A single oversized text cell must not turn the preview into an unbounded payload.
            truncated = [c for c, value in record.items() if isinstance(value, str) and len(value) > 2000]
            output.append({'id': row_id, 'flagged': row_id in flagged,
                           'values': {c: (v[:2000] + '…' if c in truncated else v) for c, v in record.items()},
                           'truncated': truncated})
        return Response({'dataset_id': str(dataset.id), 'revision': dataset.cursor, 'rows': output,
                         'page': page, 'page_size': size, 'total': count, 'pages': max(1, (count + size - 1) // size)})
