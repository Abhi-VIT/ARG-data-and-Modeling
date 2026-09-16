import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient
from workspace.ingestion import DataError
from workspace.models import Job
from workspace.deep.config import configuration
from workspace.deep.data import tabular_data, split_indices, transform
from workspace.deep.images import read_images
from workspace.deep.networks import restore_checkpoint
from workspace.deep.training import train_deep, outputs, choose_device


def frame():
    rng=np.random.default_rng(7)
    x=rng.normal(size=120);z=rng.normal(size=120)
    return pd.DataFrame({'x':x,'z':z,'y':3*x-z,'label':np.tile(['a','b'],60),'time':np.arange(120)})


def options(**changes):
    return {'features':['x','z'],'target':'y','epochs':2,'hidden_layers':[8],
            'dropout':0.,'device':'cpu','batch_size':16,'sequence_length':4,
            'recurrent_units':4,'conv_channels':[4],'latent_dim':1,**changes}


def image_files():
    rng=np.random.default_rng(3)
    result=[]
    for category in ['a','b']:
        for i in range(12):
            value=rng.integers(0,256,(16,16,3),dtype=np.uint8)
            buffer=io.BytesIO();Image.fromarray(value).save(buffer,format='PNG')
            result.append((f'collection/{category}/{i}.png',buffer.getvalue()))
    return result


def archive(files=None):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as output:
        for name,data in files if files is not None else image_files():output.writestr(name,data)
    buffer.seek(0)
    return buffer


class DeepTrainingTests(SimpleTestCase):
    def test_all_architectures_train_and_restore_best_checkpoint(self):
        for architecture in ['mlp','rnn','lstm','gru','autoencoder','cnn']:
            with self.subTest(architecture=architecture),tempfile.TemporaryDirectory() as directory:
                config=options(architecture=architecture)
                images=read_images(archive()) if architecture=='cnn' else None
                path=Path(directory)/'best.pt';epochs=[]
                result,packet,output=train_deep(config,frame=frame(),image_data=images,checkpoint_path=path,on_epoch=epochs.append)
                model,restored=restore_checkpoint(path)
                self.assertEqual(len(epochs),2);self.assertEqual(epochs[-1]['epoch'],2)
                self.assertEqual(packet['best_epoch'],restored['best_epoch'])
                for key,value in packet['state_dict'].items():torch.testing.assert_close(value,restored['state_dict'][key])
                self.assertTrue(result['figures']);json.dumps(result,allow_nan=False)
                if architecture=='cnn':
                    test_ids=split_indices(len(images[1]),configuration(config),images[1])[2]
                    test=images[0][test_ids].astype(np.float32)/255.
                else:test=tabular_data(frame(),configuration(config))[0][2][0]
                predicted=outputs(model,test,torch.device('cpu'),16)
                if architecture=='autoencoder':
                    np.testing.assert_allclose(output.reconstruction_error,((predicted-test)**2).mean(axis=1))
                    self.assertIn('latent_1',output);self.assertIn('anomaly_cutoff',restored)
                elif architecture=='cnn':
                    self.assertEqual(output.image_name.tolist(),[images[2]['names'][i] for i in test_ids])
                    np.testing.assert_allclose(output['probability [2] b'],torch.softmax(torch.tensor(predicted),dim=1).numpy()[:,1])
                else:
                    scaling=restored['preprocessing']['target_scaler']
                    np.testing.assert_allclose(output.prediction,predicted[:,0]*scaling['scale']+scaling['mean'])
                    rmse=np.sqrt(np.mean((output.actual-output.prediction)**2))
                    self.assertAlmostEqual(next(m['value'] for m in result['metrics'] if m['name']=='Test RMSE'),rmse,places=5)

    def test_optimizers_and_classification_accuracy(self):
        for optimizer,activation in [('adam','relu'),('sgd','tanh'),('rmsprop','gelu')]:
            report,packet,output=train_deep(options(task='classification',target='label',optimizer=optimizer,activation=activation),frame=frame())
            self.assertIn('validation_accuracy',packet['history'][0])
            self.assertAlmostEqual(next(m['value'] for m in report['metrics'] if m['name']=='Test accuracy'),(output.actual==output.prediction).mean())

    def test_validation_controls_early_stopping_and_failure_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'best.pt'
            # Fixed validation predictions make the first epoch the best, independently of training.
            with patch('workspace.deep.training.outputs',side_effect=lambda model,x,device,batch:np.zeros((len(x),2),dtype=np.float32)):
                _,packet,_=train_deep(options(task='classification',target='label',epochs=20,patience=2),frame=frame(),checkpoint_path=path)
            self.assertEqual(len(packet['history']),3);self.assertEqual(packet['best_epoch'],1)
            def interrupt(state):raise DataError('Simulated interruption after checkpoint save')
            with self.assertRaises(DataError):train_deep(options(),frame=frame(),checkpoint_path=path,on_epoch=interrupt)
            self.assertEqual(restore_checkpoint(path)[1]['best_epoch'],1)

    def test_training_only_preprocessing_and_chronological_windows(self):
        data=frame();data.loc[110,'x']=10000.;data.loc[0,'x']=np.nan
        c=configuration(options(architecture='lstm',order_column='time'))
        datasets,meta=tabular_data(data,c);train,val,test=split_indices(len(data),c,chronological=True)
        self.assertAlmostEqual(meta['scaler']['median'][0],data.iloc[train].x.median())
        self.assertAlmostEqual(meta['target_scaler']['mean'],data.iloc[train].y.mean(),places=5)
        for (x,y),ids,rows in zip(datasets,[train,val,test],meta['row_ids']):
            self.assertEqual(rows,(ids[4:]+1).tolist())
            np.testing.assert_allclose(x[0],transform(data.iloc[ids[:4]][['x','z']].to_numpy(),meta['scaler']))
            expected=(data.iloc[ids[4]].y-meta['target_scaler']['mean'])/meta['target_scaler']['scale']
            self.assertAlmostEqual(float(y[0]),expected,places=5)
        self.assertLess(max(meta['row_ids'][0]),min(meta['row_ids'][1]))
        self.assertLess(max(meta['row_ids'][1]),min(meta['row_ids'][2]))

    def test_invalid_settings_and_data_limits(self):
        for changes in [{'epochs':201},{'learning_rate':float('nan')},{'hidden_layers':[10000]},{'early_stopping':'yes'},{'architecture':'code'}]:
            with self.subTest(changes=changes),self.assertRaises(DataError):configuration(options(**changes))
        for changes in [{'features':['label']},{'features':['x','y']},{'features':[{}]},{'architecture':'autoencoder','latent_dim':2},{'architecture':'gru','sequence_length':30}]:
            with self.subTest(changes=changes),self.assertRaises(DataError):tabular_data(frame(),configuration(options(**changes)))
        data=frame();data.loc[0,'y']=np.nan
        with self.assertRaises(DataError):tabular_data(data,configuration(options(architecture='rnn')))
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(str(choose_device('auto')),'cpu')
            with self.assertRaises(DataError):choose_device('cuda')


class DeepImageTests(SimpleTestCase):
    def test_validated_archive_is_deduplicated_before_splitting(self):
        files=image_files();files.append(('collection/a/duplicate.png',files[0][1]))
        pixels,labels,manifest=read_images(archive(files))
        self.assertEqual(pixels.shape,(24,3,64,64));self.assertEqual(manifest['duplicates_removed'],1)
        self.assertEqual(manifest['classes'],['a','b']);self.assertEqual(len(labels),24)

    def test_unsafe_and_ambiguous_images_are_rejected(self):
        files=image_files()
        for name,data in [('../outside.png',files[0][1]),('/outside.png',files[0][1]),('collection/a/program.txt',b'code'),
                          ('collection/a/fake.png',b'invalid'),('collection/b/conflict.png',files[0][1]),(files[0][0],files[1][1])]:
            with self.subTest(name=name),self.assertRaises(DataError):read_images(archive(files+[(name,data)]))
        with self.assertRaises(DataError):read_images(archive(files[:5]))


class DeepApiTests(TestCase):
    def setUp(self):
        self.directory=Path(tempfile.mkdtemp());self.addCleanup(shutil.rmtree,self.directory)
        setting=override_settings(MEDIA_ROOT=self.directory);setting.enable();self.addCleanup(setting.disable)
        cache.clear();self.owner=get_user_model().objects.create_user('deep-owner',password='test-password')
        self.other=get_user_model().objects.create_user('deep-other',password='test-password')
        self.client=APIClient();self.client.force_login(self.owner)
        response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile('train.csv',frame().to_csv(index=False).encode(),content_type='text/csv')},format='multipart')
        self.assertEqual(response.status_code,202,response.content)
        self.state=self.client.get('/api/state/').json()['dataset']

    def train(self,**changes):
        return self.client.post('/api/deep/train/',{'dataset_id':self.state['id'],'revision_id':self.state['revision_id'],'deep':options(**changes)},format='json')

    def test_training_reports_checkpoint_download_and_owner_isolation(self):
        response=self.train();self.assertEqual(response.status_code,202,response.content)
        job=response.json();self.assertEqual(job['status'],'succeeded',job)
        self.assertEqual(len(job['result']['history']),2)
        checkpoint=io.BytesIO(b''.join(self.client.get(job['result']['checkpoint_url']).streaming_content))
        self.assertEqual(restore_checkpoint(checkpoint)[1]['source']['revision_id'],self.state['revision_id'])
        report=json.loads(b''.join(self.client.get(job['result']['report_url']).streaming_content))
        self.assertEqual(report['analysis_type'],'deep')
        csv=b''.join(self.client.get(report['assignments_url']).streaming_content)
        self.assertEqual(len(pd.read_csv(io.BytesIO(csv))),24)
        state=self.client.get('/api/state/').json();self.assertEqual(state['dataset']['revision_id'],self.state['revision_id'])
        self.assertEqual(state['deep_runs'][0]['id'],job['id'])
        # Interrupted training still exposes its last completed checkpoint to the owner.
        Job.objects.filter(pk=job['id']).update(status='failed')
        downloaded=self.client.get(job['result']['checkpoint_url']);self.assertEqual(downloaded.status_code,200);downloaded.close()
        self.client.force_login(self.other)
        for url in [job['result']['checkpoint_url'],job['result']['report_url'],report['assignments_url']]:self.assertEqual(self.client.get(url).status_code,404)

    def test_deep_queue_and_stale_revision(self):
        with override_settings(CELERY_TASK_ALWAYS_EAGER=False),patch('workspace.api.run_deep_job.apply_async') as queued:
            response=self.train();self.assertEqual(response.json()['status'],'queued')
            queued.assert_called_once_with(args=[response.json()['id']],queue='deep')
            self.assertEqual(self.train().status_code,400)
        Job.objects.filter(pk=response.json()['id']).update(status='failed')
        self.state['revision_id']=-1;self.assertEqual(self.train().status_code,400)

    def test_separate_image_zip_upload_and_cnn_without_tabular_dataset(self):
        self.client.force_login(self.other)
        response=self.client.post('/api/deep/images/',{'archive':SimpleUploadedFile('images.zip',archive().getvalue(),content_type='application/zip')},format='multipart')
        self.assertEqual(response.status_code,202,response.content);collection=response.json()
        self.assertEqual(collection['status'],'succeeded',collection)
        self.assertIsNone(self.client.get('/api/state/').json()['dataset'])
        response=self.train(architecture='cnn',image_id=collection['id']);self.assertEqual(response.status_code,202,response.content)
        self.assertEqual(response.json()['status'],'succeeded',response.content)
        self.client.force_login(self.owner)
        self.assertEqual(self.train(architecture='cnn',image_id=collection['id']).status_code,404)
        self.assertEqual(self.client.get('/api/state/').json()['image_sets'],[])
        self.assertEqual(self.train(architecture='cnn',image_id='malformed').status_code,400)

    def test_folder_upload_and_aggregate_size_limit(self):
        files=image_files()
        def upload():
            return self.client.post('/api/deep/images/',{'images':[SimpleUploadedFile(Path(name).name,data,content_type='image/png') for name,data in files],
                                                       'paths':json.dumps([name for name,_ in files])},format='multipart')
        response=upload();self.assertEqual(response.status_code,202,response.content)
        self.assertEqual(response.json()['result']['images'],24)
        with override_settings(MAX_UPLOAD_BYTES=1000):self.assertEqual(upload().status_code,400)
        self.assertEqual(self.client.get('/api/state/').json()['dataset']['revision_id'],self.state['revision_id'])
