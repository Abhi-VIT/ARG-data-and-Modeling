import json
import logging
from pathlib import Path
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.db import transaction
from django.utils import timezone
from .models import Dataset, Job, Revision, Upload, Workspace
from .ingestion import DataError, choices, fetch_csv, parse_file, validate_file, ROW_ID
from .cleaning import apply_operation, flags
from .storage import export_frame, private_path, read_frame, write_frame

logger = logging.getLogger(__name__)


def progress(job, percent, message):
    Job.objects.filter(pk=job.pk).update(progress=percent, message=message, updated_at=timezone.now())


def active_revision(job):
    dataset = Dataset.objects.get(pk=job.payload['dataset_id'], workspace=job.workspace)
    if dataset.cursor != job.payload['revision'] or job.workspace.active_dataset_id != dataset.id:
        raise DataError('The dataset changed. Refresh before trying again.')
    revision=dataset.revisions.get(number=dataset.cursor)
    if job.payload.get('revision_id',revision.pk)!=revision.pk:
        raise DataError('The source revision was replaced. Refresh before trying again.')
    return dataset, revision


def save_revision(dataset, frame, operation):
    path, summary = write_frame(dataset.workspace_id, frame)
    with transaction.atomic():
        # A new edit after undo abandons the redo branch, while retaining its private files for cleanup.
        dataset.revisions.filter(number__gt=dataset.cursor).delete()
        dataset.cursor += 1
        Revision.objects.create(dataset=dataset, number=dataset.cursor, path=path, operation=operation, profile=summary)
        dataset.save(update_fields=['cursor'])


def perform(job):
    payload = job.payload
    if job.kind == 'predict':
        perform_prediction(job)
    elif job.kind in {'ingest', 'url'}:
        progress(job, 15, 'Validating source')
        if job.kind == 'url':
            path = private_path(job.workspace_id, '.csv')
            mime = fetch_csv(payload['url'], path)
            upload = Upload.objects.create(workspace=job.workspace, name='URL import.csv', path=str(path), mime=mime)
        else:
            upload = Upload.objects.get(pk=payload['upload_id'], workspace=job.workspace)
        ext = validate_file(upload.path, upload.name, upload.mime)
        options = payload.get('options', {})
        available = choices(upload.path, ext)
        if available and not (options.get('sheets') or options.get('table')):
            job.result = {'choose': available, 'upload_id': str(upload.id)}
            return
        progress(job, 35, 'Parsing dataset')
        frame = parse_file(upload.path, upload.name, upload.mime, options)
        progress(job, 70, 'Building profile and private Parquet snapshot')
        path, summary = write_frame(job.workspace_id, frame)
        with transaction.atomic():
            dataset = Dataset.objects.create(workspace=job.workspace, name=upload.name, original_schema=list(frame.columns))
            Revision.objects.create(dataset=dataset, number=0, path=path, profile=summary,
                                    operation={'action': 'import', 'options': options})
            Workspace.objects.filter(pk=job.workspace_id).update(active_dataset=dataset)
        job.result = {'dataset_id': str(dataset.id)}
    elif job.kind in {'model', 'analysis', 'clean', 'replay', 'detect', 'export', 'undo', 'redo', 'clear'}:
        dataset, revision = active_revision(job)
        if job.kind == 'clear':
            Workspace.objects.filter(pk=job.workspace_id).update(active_dataset=None)
            return
        if job.kind in {'undo', 'redo'}:
            destination = dataset.cursor + (-1 if job.kind == 'undo' else 1)
            if not dataset.revisions.filter(number=destination).exists():
                raise DataError('No more operations to ' + job.kind + '.')
            dataset.cursor = destination
            dataset.save(update_fields=['cursor'])
            return
        progress(job, 20, 'Loading current revision')
        frame = read_frame(revision.path)
        if job.kind == 'model':
            import joblib
            from .ml.training import train
            report, bundle, output = train(frame, payload['model'], lambda percent, message: progress(job, percent, message))
            source = {'dataset_id': str(dataset.id), 'dataset_name': dataset.name, 'revision': dataset.cursor,
                      'revision_id': revision.pk, 'created_at': timezone.now().isoformat()}
            report.update(source); bundle['source'] = source
            path = private_path(job.workspace_id, '.json')
            joblib.dump(bundle, path.with_suffix('.pkl'), compress=3)
            if output is not None:
                export_frame(output, path.with_suffix('.csv'), 'csv')
                report['assignments_url'] = f'/api/models/{job.id}/output/'
            report['model_url'] = f'/api/models/{job.id}/download/'
            report['model_job_id'] = str(job.id)
            path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            job.artifact = str(path)
            job.result = {'title': report['title'], 'analysis_type': 'ml', **source,
                          'task_type': report['task_type'], 'model_key': report['model_key'], 'can_predict': report['can_predict'],
                          'model_url': report['model_url'], 'report_url': f'/api/jobs/{job.id}/report/',
                          'download_url': f'/api/jobs/{job.id}/download/', 'filename': 'model-report.json'}
        elif job.kind == 'analysis':
            from .analytics import analyze
            report = analyze(frame, payload['analysis'], lambda percent, message: progress(job, percent, message))
            report.update(dataset_id=str(dataset.id), dataset_name=dataset.name, revision=dataset.cursor, revision_id=revision.pk,
                          created_at=timezone.now().isoformat())
            path = private_path(job.workspace_id, '.json')
            path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            job.artifact = str(path)
            job.result = {'title': report['title'], 'analysis_type': report['analysis_type'],
                          'dataset_name': dataset.name,
                          'dataset_id': str(dataset.id), 'revision': dataset.cursor,
                          'report_url': f'/api/jobs/{job.id}/report/',
                          'download_url': f'/api/jobs/{job.id}/download/',
                          'filename': f'analysis-{report["analysis_type"]}-revision-{dataset.cursor}.json'}
        elif job.kind == 'detect':
            operation = payload['operation']
            if operation.get('action') not in {'detect_outliers', 'detect_duplicates'}:
                raise DataError('Choose duplicate or outlier detection.')
            mask = flags(frame, operation)
            path = private_path(job.workspace_id, '.parquet')
            import pandas as pd
            pd.DataFrame({ROW_ID: frame.index[mask]}).to_parquet(path, index=False)
            job.artifact = str(path)
            job.result = {'flagged_count': int(mask.sum()), 'dataset_id': str(dataset.id), 'revision': dataset.cursor,
                          'operation': operation}
        elif job.kind == 'clean':
            operation = payload['operation']
            if operation.get('action') == 'remove_outliers':
                detection = Job.objects.filter(pk=payload.get('detection_job'), workspace=job.workspace,
                                               kind='detect', status='succeeded').first()
                if not detection or detection.result.get('dataset_id') != str(dataset.id) or detection.result.get('revision') != dataset.cursor:
                    raise DataError('Preview outliers in this revision before confirming their removal.')
                operation = dict(detection.result['operation'], action='remove_outliers')
                if detection.result['operation'].get('action') != 'detect_outliers':
                    raise DataError('Preview outliers before removing them.')
            progress(job, 45, 'Applying cleaning operation')
            cleaned = apply_operation(frame, operation)
            progress(job, 80, 'Saving reversible revision')
            save_revision(dataset, cleaned, operation)
        elif job.kind == 'replay':
            recipe = payload['recipe']
            if not isinstance(recipe, dict) or recipe.get('version') != 1 or recipe.get('schema') != list(frame.columns):
                raise DataError('Recipe version or ordered input columns do not match this dataset.')
            operations = recipe.get('operations')
            if not isinstance(operations, list) or not 1 <= len(operations) <= 100:
                raise DataError('Recipes must contain 1–100 operations.')
            # Compute the whole recipe before changing any database state.
            staged = []
            for i, operation in enumerate(operations):
                frame = apply_operation(frame, operation)
                staged.append((write_frame(job.workspace_id, frame), operation))
                progress(job, 25 + int(65 * (i + 1) / len(operations)), f'Replaying step {i+1} of {len(operations)}')
            with transaction.atomic():
                dataset.revisions.filter(number__gt=dataset.cursor).delete()
                for (path, summary), operation in staged:
                    dataset.cursor += 1
                    Revision.objects.create(dataset=dataset, number=dataset.cursor, path=path, profile=summary, operation=operation)
                dataset.save(update_fields=['cursor'])
        else:
            format = payload['format']
            if format not in {'csv', 'tsv', 'json', 'parquet', 'xlsx'}:
                raise DataError('Unsupported export format.')
            path = private_path(job.workspace_id, '.' + format)
            export_frame(frame, path, format)
            job.artifact = str(path)
            job.result = {'filename': Path(dataset.name).stem + '-cleaned.' + format,
                          'download_url': f'/api/jobs/{job.id}/download/'}
    else:
        raise DataError('Unsupported job type.')


def perform_prediction(job):
    import joblib
    from .ml.prediction import predict
    from .analytics.common import report, dataframe_table, clean_json
    source = Job.objects.get(pk=job.payload['model_id'], workspace=job.workspace, kind='model', status='succeeded')
    upload = Upload.objects.get(pk=job.payload['upload_id'], workspace=job.workspace)
    model_path = Path(source.artifact).with_suffix('.pkl')
    if not model_path.is_file():raise DataError('Saved model file is no longer available. Train the model again.')
    progress(job, 20, 'Parsing the prediction upload')
    ext=validate_file(upload.path, upload.name, upload.mime)
    options=job.payload.get('options',{})
    available=choices(upload.path,ext)
    if available and not (options.get('sheets') or options.get('table')):
        raise DataError('Specify worksheet names or a table in prediction parsing options, or upload CSV/Parquet.')
    frame=parse_file(upload.path, upload.name, upload.mime, options)
    # Only server-created artifacts belonging to this workspace are deserialized.
    bundle=joblib.load(model_path)
    progress(job, 65, 'Scoring new observations with the fitted preprocessing pipeline')
    output=predict(bundle,frame,job.payload.get('threshold',.5))
    path=private_path(job.workspace_id,'.json')
    export_frame(output,path.with_suffix('.csv'),'csv')
    result=report('Predictions · '+upload.name,'prediction')
    result.update(bundle['source']);result['created_at']=timezone.now().isoformat()
    result['selection']={'used':list(bundle['schema'])};result['model_job_id']=str(source.id)
    result['metrics']=[{'name':'Rows scored','value':len(output)}]
    result['tables']=[dataframe_table('Prediction preview (first 30 rows)',output.head(30))]
    result['notes']=['Predictions follow uploaded row order. Extra columns are ignored; required features are validated. Your active dataset has not been replaced.',
                     'Preprocessing is reused from training and is never refitted on the prediction upload.']
    if bundle['classes'] and len(bundle['classes'])==2:
        result['notes'].append(f'Positive class: {bundle["classes"][1]}; probability threshold: {job.payload.get("threshold",.5)}.')
    result['assignments_url']=f'/api/models/{job.id}/output/'
    path.write_text(json.dumps(clean_json(result),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    job.artifact=str(path)
    job.result={'title':result['title'],'analysis_type':'prediction','dataset_name':bundle['source']['dataset_name'],
                'dataset_id':bundle['source']['dataset_id'],'revision':bundle['source']['revision'],
                'report_url':f'/api/jobs/{job.id}/report/','download_url':f'/api/jobs/{job.id}/download/',
                'filename':'prediction-report.json','output_url':result['assignments_url']}


@shared_task
def run_job(job_id):
    # Duplicate broker delivery must not apply an operation twice.
    claimed = Job.objects.filter(pk=job_id, status='queued').update(status='running', progress=5, updated_at=timezone.now())
    if not claimed:
        return
    job = Job.objects.select_related('workspace').get(pk=job_id)
    try:
        perform(job)
        job.status, job.progress, job.message = 'succeeded', 100, 'Complete'
    except (DataError, SoftTimeLimitExceeded, TimeoutError) as exc:
        job.status, job.message = 'failed', str(exc)[:500] or 'Operation exceeded its time limit. Use a smaller dataset.'
    except Exception:
        logger.exception('Data job %s failed', job.id)
        job.status = 'failed'
        job.message = 'Could not process this data. Check the format, parsing options, selected columns and values.'
    job.save(update_fields=['status', 'progress', 'message', 'result', 'artifact', 'updated_at'])
