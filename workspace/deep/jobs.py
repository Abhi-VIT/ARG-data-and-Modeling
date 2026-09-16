import json
from pathlib import Path
import numpy as np
from django.utils import timezone
from ..models import Job,Upload
from ..storage import private_path,read_frame,export_frame
from ..ingestion import DataError


def perform_deep(job):
    from ..tasks import active_revision,progress
    if job.kind=='image_ingest':
        from .images import read_images
        upload=Upload.objects.get(pk=job.payload['upload_id'],workspace=job.workspace)
        pixels,labels,manifest=read_images(upload.path,lambda p,m:progress(job,p,m))
        path=private_path(job.workspace_id,'.json')
        np.savez_compressed(path.with_suffix('.npz'),pixels=pixels,labels=labels)
        path.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
        job.artifact=str(path)
        job.result={'title':upload.name,'name':upload.name,**{k:manifest[k] for k in ['images','classes','duplicates_removed','image_size']}}
        return
    from .training import train_deep
    options=job.payload['deep'];frame=None;image_data=None
    if options.get('architecture')=='cnn':
        collection=Job.objects.get(pk=options.get('image_id'),workspace=job.workspace,kind='image_ingest',status='succeeded')
        manifest=json.loads(Path(collection.artifact).read_text(encoding='utf-8'))
        with np.load(Path(collection.artifact).with_suffix('.npz'),allow_pickle=False) as arrays:
            image_data=(arrays['pixels'],arrays['labels'],manifest)
        source={'dataset_id':None,'dataset_name':collection.result['name'],'revision':0,'revision_id':None,
                'source_type':'images','image_job_id':str(collection.id)}
    else:
        dataset,revision=active_revision(job);frame=read_frame(revision.path)
        source={'dataset_id':str(dataset.id),'dataset_name':dataset.name,'revision':dataset.cursor,'revision_id':revision.pk,'source_type':'tabular'}
    source['created_at']=timezone.now().isoformat()
    path=private_path(job.workspace_id,'.json');job.artifact=str(path)
    job.save(update_fields=['artifact'])
    def epoch(state):
        job.result={**source,**state,'checkpoint_url':f'/api/deep/{job.id}/checkpoint/'}
        Job.objects.filter(pk=job.id).update(result=job.result,progress=10+int(80*state['epoch']/state['epochs']),
            message=f'Epoch {state["epoch"]}/{state["epochs"]} · validation loss {state["history"][-1]["validation_loss"]:.5g}',updated_at=timezone.now())
    progress(job,10,'Preparing neural network and train/validation/test splits')
    result,packet,output=train_deep(options,frame=frame,image_data=image_data,checkpoint_path=path.with_suffix('.pt'),on_epoch=epoch,source=source)
    result.update(source,checkpoint_url=f'/api/deep/{job.id}/checkpoint/',assignments_url=f'/api/deep/{job.id}/output/')
    export_frame(output,path.with_suffix('.csv'),'csv')
    path.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    job.result.update(title=result['title'],analysis_type='deep',report_url=f'/api/jobs/{job.id}/report/',
                      download_url=f'/api/jobs/{job.id}/download/',filename='deep-learning-report.json')
