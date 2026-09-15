import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient
from sklearn.model_selection import train_test_split
from workspace.ingestion import DataError
from workspace.ml.catalog import CATALOG, validate_params
from workspace.ml.training import train
from workspace.ml.prediction import predict
from workspace.ml.evaluation import threshold_counts
from workspace.models import Job


def sample_frame():
    rng=np.random.default_rng(22)
    x=rng.normal(size=100);z=rng.normal(size=100)
    return pd.DataFrame({'x':x,'z':z,'category':np.tile(['A','B'],50),'y':4+2*x-.4*z+rng.normal(0,.1,100),'label':np.where(x+z>0,'positive','negative')})


class ModelTests(SimpleTestCase):
    def test_every_catalog_estimator_fits_and_serializes(self):
        frame=sample_frame()
        for key,spec in CATALOG.items():
            with self.subTest(model=key):
                params={}
                if 'n_estimators' in spec['params']:params['n_estimators']=10
                if key=='tsne':params={'perplexity':5.,'max_iter':250}
                if key=='umap':params={'n_neighbors':5}
                options={'model':key,'features':['x','z','category'],'target':'label' if spec['task']=='classification' else 'y','params':params}
                result,bundle,output=train(frame,options)
                json.dumps(result,allow_nan=False)
                self.assertTrue(result['figures'])
                buffer=io.BytesIO();joblib.dump(bundle,buffer);buffer.seek(0);restored=joblib.load(buffer)
                if bundle['can_predict']:
                    scored=predict(restored,frame.iloc[:4])
                    self.assertEqual(len(scored),4)
                else:
                    self.assertIsNotNone(output)
                    with self.assertRaises(DataError):predict(restored,frame.iloc[:4])

    def test_preprocessing_fits_training_only_and_survives_new_categories(self):
        frame=sample_frame();frame.loc[0,'x']=10000.;frame.loc[1,'x']=np.nan
        result,bundle,_=train(frame,{'model':'ridge','features':['x','category'],'target':'y','seed':42})
        xtrain,xtest=train_test_split(frame[['x','category']],test_size=.2,random_state=42)
        imputer=bundle['pipeline'].named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
        self.assertAlmostEqual(imputer.statistics_[0],xtrain.x.median())
        new=pd.DataFrame({'x':[1,np.nan],'category':['never-seen',None],'irrelevant':[1,2]})
        output=predict(bundle,new)
        self.assertTrue(np.isfinite(output.prediction).all())
        self.assertEqual(len(output),2)
        with self.assertRaises(DataError):predict(bundle,new.drop(columns='x'))
        with self.assertRaises(DataError):predict(bundle,new.assign(x=['bad','text']))

    def test_grid_random_search_and_parameter_limits(self):
        frame=sample_frame()
        for method in ['grid','random']:
            result,bundle,_=train(frame,{'model':'ridge','features':['x','z'],'target':'y','search':{'method':method,'ranges':{'alpha':[.1,1,10]},'folds':3,'iterations':2}})
            rows=next(t['rows'] for t in result['tables'] if t['title'].startswith('Cross-validation'))
            self.assertEqual(len(rows),3 if method=='grid' else 2)
            self.assertIn(result['best_params']['alpha'],[.1,1,10])
        for params in [{'alpha':-1},{'alpha':float('nan')},{'__class__':'anything'}]:
            with self.assertRaises(DataError):validate_params('ridge',params)
        with self.assertRaises(DataError):
            train(frame,{'model':'forest_reg','features':['x'],'target':'y','search':{'method':'grid','ranges':{'n_estimators':[10,20,30,40,50],'max_depth':[1,2,3,4,5]}}})

    def test_thresholds_match_counts_and_prediction_cutoff(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.6,.4,.9])
        values=threshold_counts(y,p,.5)
        self.assertEqual([values[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        result,bundle,_=train(sample_frame(),{'model':'logistic','features':['x','z'],'target':'label'})
        self.assertEqual(len(result['thresholds']),101)
        self.assertEqual(result['thresholds'][0]['tn'],0)
        scored=predict(bundle,sample_frame(),0)
        self.assertEqual(set(scored.prediction),{bundle['classes'][1]})

    def test_search_refits_preprocessing_only_within_training_folds(self):
        from sklearn.impute import SimpleImputer
        original=SimpleImputer.fit;seen=[]
        def record_fit(instance,x,y=None,**kwargs):
            seen.append(len(x))
            return original(instance,x,y,**kwargs)
        with patch.object(SimpleImputer,'fit',record_fit):
            train(sample_frame(),{'model':'ridge','features':['x'],'target':'y','search':{'method':'grid','ranges':{'alpha':[.1,1]},'folds':3}})
        self.assertEqual(len(seen),7)
        self.assertEqual(seen.count(80),1)
        self.assertTrue(all(n in {53,54,80} for n in seen))

    def test_regression_metrics_use_the_held_out_rows(self):
        from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score
        frame=sample_frame()
        report,bundle,_=train(frame,{'model':'ridge','features':['x','z'],'target':'y','seed':7,'test_size':.3})
        _,test=train_test_split(frame,test_size=.3,random_state=7)
        predicted=predict(bundle,test).prediction.to_numpy()
        values={m['name']:m['value'] for m in report['metrics']}
        for name,fn in [('RMSE',root_mean_squared_error),('MAE',mean_absolute_error),('R²',r2_score)]:
            self.assertAlmostEqual(values[name],fn(test.y,predicted))

    def test_multiclass_metrics_and_target_guards(self):
        frame=sample_frame();frame['label']=np.tile(['a','b','c','d'],25)
        result,_,_=train(frame,{'model':'logistic','features':['x','z'],'target':'label'})
        self.assertNotIn('thresholds',result)
        self.assertIn('ROC-AUC',[m['name'] for m in result['metrics']])
        for options in [{'model':'ridge','features':['x','y'],'target':'y'}, {'model':'ridge','features':['x'],'target':'label'}, {'model':'ridge','features':[],'target':'y'}]:
            with self.assertRaises(DataError):train(frame,options)

    def test_unsupervised_diagnostics_and_sample_guards(self):
        frame=sample_frame()
        result,_,output=train(frame,{'model':'kmeans','features':['x','z']})
        self.assertEqual(len(output),len(frame))
        self.assertTrue(any(f['title']=='Elbow curve' for f in result['figures']))
        result,_,_=train(frame,{'model':'hierarchical','features':['x','z']})
        self.assertTrue(any('Dendrogram' in f['title'] for f in result['figures']))
        result,_,_=train(frame,{'model':'pca','features':['x','z']})
        self.assertAlmostEqual(next(m['value'] for m in result['metrics'] if m['name']=='Explained variance retained'),1.)
        for options in [{'model':'pca','features':['x'],'params':{'n_components':2}}, {'model':'tsne','features':['x','z'],'params':{'perplexity':100.}}]:
            with self.assertRaises(DataError):train(frame,options)


class ModelApiTests(TestCase):
    def setUp(self):
        self.directory=Path(tempfile.mkdtemp());self.addCleanup(shutil.rmtree,self.directory)
        setting=override_settings(MEDIA_ROOT=self.directory);setting.enable();self.addCleanup(setting.disable)
        cache.clear()
        self.owner=get_user_model().objects.create_user('model-owner',password='test-password')
        self.other=get_user_model().objects.create_user('model-other',password='test-password')
        self.client=APIClient();self.client.force_login(self.owner)
        response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile('train.csv',sample_frame().to_csv(index=False).encode(),content_type='text/csv')},format='multipart')
        self.assertEqual(response.status_code,202,response.content)
        self.state=self.client.get('/api/state/').json()['dataset']

    def fit(self):
        response=self.client.post('/api/jobs/',{'kind':'model','dataset_id':self.state['id'],'revision':self.state['revision'],'model':{'model':'ridge','features':['x','z','category'],'target':'y'}},format='json')
        self.assertEqual(response.status_code,202,response.content)
        self.assertEqual(response.json()['status'],'succeeded',response.content)
        return response.json()

    def test_train_download_predict_and_owner_isolation(self):
        model=self.fit()
        report=json.loads(b''.join(self.client.get(model['result']['report_url']).streaming_content))
        self.assertEqual(report['revision_id'],self.state['revision_id'])
        artifact=b''.join(self.client.get(model['result']['model_url']).streaming_content)
        self.assertEqual(joblib.load(io.BytesIO(artifact))['schema'],{'x':'numeric','z':'numeric','category':'categorical'})
        response=self.client.post(f'/api/models/{model["id"]}/predict/',{'file':SimpleUploadedFile('new.csv',b'x,z,category\n1,2,never-seen\n,3,A\n',content_type='text/csv')},format='multipart')
        self.assertEqual(response.status_code,202,response.content)
        result=response.json()
        output=b''.join(self.client.get(result['result']['output_url']).streaming_content)
        self.assertEqual(len(pd.read_csv(io.BytesIO(output))),2)
        self.assertEqual(self.client.get('/api/state/').json()['dataset']['revision_id'],self.state['revision_id'])
        self.client.force_login(self.other)
        for url in [model['result']['model_url'],model['result']['report_url'],result['result']['output_url']]:self.assertEqual(self.client.get(url).status_code,404)
        self.assertEqual(self.client.post(f'/api/models/{model["id"]}/predict/',{},format='multipart').status_code,404)

    def test_prediction_rejects_serialized_upload_and_missing_schema(self):
        model=self.fit();url=f'/api/models/{model["id"]}/predict/'
        self.assertEqual(self.client.post(url,{'file':SimpleUploadedFile('model.pkl',b'anything')},format='multipart').status_code,400)
        response=self.client.post(url,{'file':SimpleUploadedFile('new.csv',b'wrong\n1\n',content_type='text/csv')},format='multipart')
        self.assertEqual(response.json()['status'],'failed')
        self.assertIn('missing required features',response.json()['message'])

    def test_training_queues_and_stale_branch_is_rejected(self):
        with override_settings(CELERY_TASK_ALWAYS_EAGER=False),patch('workspace.api.run_job.delay') as delay:
            response=self.client.post('/api/jobs/',{'kind':'model','dataset_id':self.state['id'],'revision':0,'model':{'model':'ridge','features':['x'],'target':'y'}},format='json')
            self.assertEqual(response.json()['status'],'queued');delay.assert_called_once()
        from workspace.tasks import run_job
        run_job(response.json()['id'])
        response=self.client.post('/api/jobs/',{'kind':'model','dataset_id':self.state['id'],'revision':0,'revision_id':-1,'model':{'model':'ridge','features':['x'],'target':'y'}},format='json')
        self.assertEqual(response.status_code,400)
