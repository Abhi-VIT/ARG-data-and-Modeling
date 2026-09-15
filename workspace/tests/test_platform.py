import io
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, SimpleTestCase, override_settings
from rest_framework.test import APIClient

from workspace.cleaning import apply_operation, flags
from workspace.ingestion import DataError, fetch_csv, parse_file, validate_file, choices
from workspace.models import Job, Workspace
from workspace.storage import read_frame


class ParserTests(SimpleTestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

    def file(self, name, content):
        path = self.directory / name
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        return path

    def test_csv_tsv_encoding_no_header_and_regex(self):
        path = self.file('sample.csv', 'city;value\nDelhi;3\nMumbai;\n')
        frame = parse_file(path, path.name, 'text/csv', {})
        self.assertEqual(list(frame.columns), ['city', 'value'])
        self.assertTrue(pd.isna(frame.iloc[1]['value']))
        path = self.file('sample.tsv', 'a\tb\n1\t2\n')
        self.assertEqual(parse_file(path, path.name, 'text/plain', {}).iloc[0].tolist(), [1, 2])
        path = self.file('sample.txt', b'caf\xe9|4\n')
        frame = parse_file(path, path.name, 'text/plain', {'header': False, 'encoding': 'latin-1', 'delimiter': '|'})
        self.assertEqual(frame.iloc[0, 0], 'café')
        path = self.file('sample.txt', 'a   b\n1   2\n')
        frame = parse_file(path, path.name, 'text/plain', {'delimiter': r'\s+', 'regex_delimiter': True})
        self.assertEqual(frame.iloc[0].tolist(), [1, 2])

    def test_nested_json_depth_and_wrapped_records(self):
        path = self.file('sample.json', '{"records":[{"a":{"b":1},"c":[1,2]}]}')
        frame = parse_file(path, path.name, 'application/json', {'record_key': 'records', 'depth': 1})
        self.assertIn('a.b', frame.columns)
        self.assertEqual(frame.iloc[0]['c'], '[1, 2]')
        shallow = parse_file(path, path.name, '', {'record_key': 'records', 'depth': 0})
        self.assertEqual(shallow.iloc[0]['a'], '{"b": 1}')

    def test_excel_multi_sheet_parquet_and_xml(self):
        frame = pd.DataFrame({'id': [1, 2], 'score': [3.0, np.nan]})
        path = self.directory / 'sample.xlsx'
        with pd.ExcelWriter(path) as book:
            frame.to_excel(book, sheet_name='First', index=False)
            frame.to_excel(book, sheet_name='Second', index=False)
        self.assertEqual(choices(path, '.xlsx')['names'], ['First', 'Second'])
        result = parse_file(path, path.name, '', {'sheets': ['First', 'Second']})
        self.assertEqual(len(result), 4)
        self.assertIn('_source_sheet', result.columns)
        path = self.directory / 'sample.parquet'
        frame.to_parquet(path)
        pd.testing.assert_frame_equal(parse_file(path, path.name, '', {}), frame)
        path = self.file('sample.xml', '<rows><row><a>3</a><b>x</b></row></rows>')
        self.assertEqual(parse_file(path, path.name, 'application/xml', {'xpath': './row'}).iloc[0]['a'], 3)

    def test_sql_is_literal_parser_and_sqlite_is_readonly(self):
        path = self.file('sample.sql', "CREATE TABLE samples (id INT, label TEXT); INSERT INTO samples VALUES (1, 'A'), (2, 'B');")
        self.assertEqual(choices(path, '.sql')['names'], ['samples'])
        self.assertEqual(parse_file(path, path.name, '', {'table': 'samples'})['id'].tolist(), [1, 2])
        for sql in ["DROP TABLE samples;", "ATTACH DATABASE '/etc/passwd' AS x;", "SELECT load_extension('evil');",
                    'CREATE TABLE x AS SELECT 1;', 'CREATE TABLE x(a TEXT); INSERT INTO x VALUES (readfile("secret"));']:
            path = self.file('unsafe.sql', sql)
            with self.assertRaises(DataError):
                parse_file(path, path.name, '', {'table': 'x'})
        path = self.directory / 'sample.db'
        connection = sqlite3.connect(path)
        connection.execute('CREATE TABLE "quoted name" (value INT)')
        connection.execute('INSERT INTO "quoted name" VALUES (42)')
        connection.commit(); connection.close()
        self.assertEqual(parse_file(path, path.name, '', {'table': 'quoted name'}).iloc[0, 0], 42)

    def test_disguised_uploads_size_limits_and_xml_entities(self):
        for name, content, mime in [('fake.csv', b'MZexecutable', 'text/csv'),
                                    ('fake.xlsx', b'plain text', ''),
                                    ('real.csv', b'a,b\n1,2', 'application/pdf')]:
            path = self.file(name, content)
            with self.assertRaises(DataError): validate_file(path, name, mime)
        path = self.file('sample.csv', 'a,b\n1,2\n3,4\n')
        with override_settings(MAX_ROWS=1), self.assertRaises(DataError):
            parse_file(path, path.name, '', {})
        with override_settings(MAX_UPLOAD_BYTES=2), self.assertRaises(DataError):
            validate_file(path, path.name, '')
        path = self.file('evil.xml', '<!DOCTYPE foo [<!ENTITY x SYSTEM "file:///etc/passwd">]><rows><row>&x;</row></rows>')
        from defusedxml.common import DefusedXmlException
        with self.assertRaises(DefusedXmlException): parse_file(path, path.name, '', {})

    @override_settings(IMPORT_URL_HOSTS=['example.com'])
    def test_url_blocks_private_dns_credentials_and_unapproved_hosts(self):
        for url in ['http://example.com/a.csv', 'https://user:pass@example.com/a.csv', 'https://localhost/a.csv', 'https://example.com:444/a.csv']:
            with self.assertRaises(DataError): fetch_csv(url, self.directory / 'download.csv')
        with patch('workspace.ingestion.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 443))]), self.assertRaises(DataError):
            fetch_csv('https://example.com/a.csv', self.directory / 'download.csv')

    @override_settings(IMPORT_URL_HOSTS=['example.com'], MAX_UPLOAD_BYTES=20)
    def test_url_download_pins_validated_address_and_caps_stream(self):
        response = MagicMock(status=200)
        response.headers = {'Content-Type': 'text/csv'}
        response.stream.return_value = [b'a\n1\n']
        with patch('workspace.ingestion.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('93.184.216.34', 443))]), \
             patch('workspace.ingestion.urllib3.HTTPSConnectionPool') as pool:
            pool.return_value.request.return_value = response
            target = self.directory / 'remote.csv'
            self.assertEqual(fetch_csv('https://example.com/data.csv', target), 'text/csv')
            self.assertEqual(target.read_bytes(), b'a\n1\n')
            self.assertEqual(pool.call_args.args[0], '93.184.216.34')
            self.assertEqual(pool.call_args.kwargs['assert_hostname'], 'example.com')
            self.assertFalse(pool.return_value.request.call_args.kwargs['redirect'])
            response.stream.return_value = [b'x' * 21]
            with self.assertRaises(DataError): fetch_csv('https://example.com/data.csv', target)
            response.status = 302
            with self.assertRaises(DataError): fetch_csv('https://example.com/data.csv', target)

    def test_reserved_columns_are_rejected(self):
        path = self.file('reserved.csv', '__arg_rowid__,value\n1,2\n')
        with self.assertRaises(DataError): parse_file(path, path.name, 'text/csv', {})


class CleaningTests(SimpleTestCase):
    def test_fill_respects_cells_rows_and_range(self):
        frame = pd.DataFrame({'a': [1., np.nan, np.nan, 7.], 'b': [2., np.nan, 6., 8.]})
        result = apply_operation(frame, {'action':'fill','method':'mean','selection':{'columns':['a'],'rows':[1]}})
        self.assertEqual(result.loc[1,'a'], 4.)
        self.assertTrue(pd.isna(result.loc[2,'a']))
        self.assertTrue(pd.isna(result.loc[1,'b']))
        for method in ['median','mode','ffill','bfill','interpolate','knn']:
            with self.subTest(method=method):
                result = apply_operation(frame, {'action':'fill','method':method,'selection':{'columns':['a']}})
                self.assertFalse(result['a'].isna().any())
        result = apply_operation(frame, {'action':'fill','method':'constant','value':0,'selection':{'columns':['a'],'range':[1,2]}})
        self.assertEqual(result['a'].tolist(), [1,0,0,7])

    def test_group_fill_drop_missing_and_duplicates(self):
        frame = pd.DataFrame({'group':['a','a','b','b'], 'value':[2.,np.nan,5.,np.nan]})
        result=apply_operation(frame, {'action':'fill','method':'group_mode','group':'group','selection':{'columns':['value']}})
        self.assertEqual(result['value'].tolist(), [2,2,5,5])
        self.assertEqual(len(apply_operation(frame, {'action':'drop_missing','method':'any'})), 2)
        result=apply_operation(frame, {'action':'drop_columns','threshold':40})
        self.assertEqual(list(result.columns), ['group'])
        repeated=pd.concat([frame,frame],ignore_index=True)
        self.assertEqual(len(apply_operation(repeated,{'action':'drop_duplicates'})),4)

    def test_nullable_integer_fill_promotes_fractional_mean(self):
        frame = pd.DataFrame({'a': pd.Series([1, None, 2], dtype='Int64')})
        result = apply_operation(frame, {'action': 'fill', 'method': 'mean'})
        self.assertEqual(result.loc[1, 'a'], 1.5)

    def test_outliers_all_methods_and_no_silent_removal(self):
        frame=pd.DataFrame({'value':[1.,2.,2.,3.,2.,3.,2.,2.,100.]})
        for method in ['iqr','zscore','isolation_forest']:
            with self.subTest(method=method):
                mask=flags(frame,{'action':'detect_outliers','method':method,'threshold':2,'contamination':.1})
                self.assertTrue(mask.iloc[-1])
                self.assertEqual(len(frame),9)

    def test_text_conversion_encoding_scaling_and_column_operations(self):
        frame=pd.DataFrame({'text':[' Low! ','HIGH'], 'a':[1.,3.], 'b':[2.,4.]})
        for method in ['trim','lower','upper','title','special','regex']:
            with self.subTest(method=method):
                result=apply_operation(frame, {'action':'text','method':method,'pattern':'!', 'replacement':'', 'selection':{'columns':['text']}})
                self.assertEqual(result['a'].tolist(),[1,3])
        result=apply_operation(frame,{'action':'rename','name':'score','selection':{'columns':['a']}})
        self.assertIn('score',result.columns)
        result=apply_operation(frame,{'action':'merge','name':'both','delimiter':'|','selection':{'columns':['a','b']}})
        self.assertEqual(result.iloc[0]['both'],'1.0|2.0')
        result=apply_operation(result,{'action':'split','delimiter':'|','selection':{'columns':['both']}})
        self.assertIn('both_2',result.columns)
        for method in ['standard','minmax','robust','log']:
            self.assertEqual(len(apply_operation(frame,{'action':'scale','method':method,'selection':{'columns':['a','b']}})),2)
        for method in ['label','onehot','ordinal','target']:
            result=apply_operation(frame,{'action':'encode','method':method,'order':[' Low! ','HIGH'],'target':'a','selection':{'columns':['text']}})
            self.assertEqual(len(result),2)
        result=apply_operation(frame,{'action':'convert','method':'string','selection':{'columns':['a']}})
        self.assertEqual(result.iloc[0]['a'],'1.0')
        dates=pd.DataFrame({'date':['31/12/2025','01/01/2026']})
        result=apply_operation(dates,{'action':'convert','method':'datetime','format':'%d/%m/%Y'})
        self.assertEqual(result.iloc[0,0].year,2025)

    def test_reject_bad_selection_regex_numeric_and_ordinal_inputs(self):
        frame=pd.DataFrame({'a':['x','y'],'b':[1.,2.]})
        for operation in [
            {'action':'fill','method':'mean'},
            {'action':'rename','name':'z','selection':{'columns':['a'],'rows':[0]}},
            {'action':'edit','selection':{'columns':['a'],'rows':[99]},'value':0},
            {'action':'encode','method':'ordinal','order':['x'],'selection':{'columns':['a']}},
            {'action':'text','method':'regex','pattern':'x'*201,'selection':{'columns':['a']}},
        ]:
            with self.subTest(operation=operation), self.assertRaises(DataError):apply_operation(frame,operation)


class WorkflowTests(TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree,self.directory)
        self.override = override_settings(MEDIA_ROOT=Path(self.directory))
        self.override.enable();self.addCleanup(self.override.disable)
        cache.clear()
        self.user=get_user_model().objects.create_user('analyst',password='test-password')
        self.other=get_user_model().objects.create_user('other',password='test-password')
        self.client=APIClient();self.client.force_login(self.user)

    def upload(self,content=b'a,b\n1,hello\n,world\n3,hello\n',name='sample.csv',mime='text/csv'):
        response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile(name,content,content_type=mime),'options':'{}'},format='multipart')
        self.assertEqual(response.status_code,202,response.content)
        return response.json()

    def state(self):return self.client.get('/api/state/').json()['dataset']

    def job(self,kind,**payload):
        state=self.state()
        response=self.client.post('/api/jobs/',{'kind':kind,'dataset_id':state['id'],'revision':state['revision'],**payload},format='json')
        self.assertEqual(response.status_code,202,response.content)
        self.assertEqual(response.json()['status'],'succeeded',response.content)
        return response.json()

    def test_import_preview_edit_undo_redo_branch_and_recipe(self):
        self.upload()
        preview=self.client.get('/api/preview/',{'q':'world'}).json()
        self.assertEqual(preview['total'],1);self.assertEqual(preview['rows'][0]['id'],1)
        self.job('clean',operation={'action':'fill','method':'mean','selection':{'columns':['a']}})
        self.assertEqual(self.state()['profile']['missing'],0)
        recipe=self.client.get('/api/recipe/').json()
        self.assertEqual(len(recipe['operations']),1)
        self.job('undo');self.assertEqual(self.state()['profile']['missing'],1)
        self.job('redo');self.assertEqual(self.state()['profile']['missing'],0)
        self.job('undo')
        self.job('clean',operation={'action':'edit','selection':{'columns':['a'],'rows':[1]},'value':8})
        self.assertFalse(self.state()['can_redo'])
        self.upload()
        self.job('replay',recipe=recipe)
        self.assertEqual(self.state()['profile']['missing'],0)

    def test_drop_columns_undo_redo_and_recipe_replay(self):
        self.upload()
        self.job('clean',operation={'action':'drop_selected_columns','selection':{'columns':['b']}})
        self.assertEqual([c['name'] for c in self.state()['profile']['columns']],['a'])
        recipe=self.client.get('/api/recipe/').json()
        self.job('undo');self.assertEqual(len(self.state()['profile']['columns']),2)
        self.job('redo');self.assertEqual(len(self.state()['profile']['columns']),1)
        self.upload();self.job('replay',recipe=recipe)
        self.assertEqual([c['name'] for c in self.state()['profile']['columns']],['a'])

    def test_analysis_reports_are_revision_scoped_persistent_and_private(self):
        self.upload(b'a,b\n1,2\n2,5\n3,6\n4,9\n5,11\n6,10\n')
        state=self.state()
        report=self.job('analysis',analysis={'type':'regression','columns':['a','b'],'target':'b','model':'simple'})
        self.assertEqual(self.state()['revision'],state['revision'])
        response=self.client.get(report['result']['report_url'])
        content=b''.join(response.streaming_content)
        result=json.loads(content)
        self.assertEqual(result['revision'],0)
        self.assertEqual(len(result['figures']),5)
        self.assertNotIn('figures',report['result'])
        self.assertEqual(b''.join(self.client.get(report['result']['download_url']).streaming_content),content)
        self.job('clean',operation={'action':'drop_selected_columns','selection':{'columns':['a']}})
        self.assertEqual(b''.join(self.client.get(report['result']['report_url']).streaming_content),content)
        self.assertEqual(self.client.get('/api/state/').json()['analyses'][0]['id'],report['id'])
        self.client.force_login(self.other)
        self.assertEqual(self.client.get(report['result']['report_url']).status_code,404)
        self.assertEqual(self.client.get(report['result']['download_url']).status_code,404)
        self.assertEqual(self.client.get('/api/state/').json()['analyses'],[])

    def test_report_distinguishes_replaced_revision_after_undo(self):
        self.upload()
        self.job('clean',operation={'action':'fill','method':'constant','value':2,'selection':{'columns':['a']}})
        report=self.job('analysis',analysis={'type':'descriptive','columns':['a']})
        saved=json.loads(b''.join(self.client.get(report['result']['report_url']).streaming_content))
        self.job('undo')
        self.job('clean',operation={'action':'fill','method':'constant','value':20,'selection':{'columns':['a']}})
        self.assertEqual(saved['revision'],self.state()['revision'])
        self.assertNotEqual(saved['revision_id'],self.state()['revision_id'])

    @override_settings(CELERY_TASK_ALWAYS_EAGER=False)
    def test_analysis_is_queued_and_stale_selection_rejected(self):
        with override_settings(CELERY_TASK_ALWAYS_EAGER=True):self.upload()
        state=self.state()
        payload={'kind':'analysis','dataset_id':state['id'],'revision':0,'analysis':{'type':'descriptive','columns':['a']}}
        with patch('workspace.api.run_job.delay') as enqueue:
            response=self.client.post('/api/jobs/',payload,format='json')
            self.assertEqual(response.json()['status'],'queued')
            enqueue.assert_called_once()
        from workspace.tasks import run_job
        run_job(response.json()['id'])
        self.assertEqual(Job.objects.get(pk=response.json()['id']).status,'succeeded')
        payload['revision']=99
        self.assertEqual(self.client.post('/api/jobs/',payload,format='json').status_code,400)
        payload['revision']=0;payload['analysis']['columns']=[]
        self.assertEqual(self.client.post('/api/jobs/',payload,format='json').status_code,400)

    def test_export_formats_and_formula_safety(self):
        self.upload(b'a,b\n1,=1+1\n2,test\n')
        for format in ['csv','tsv','json','parquet','xlsx']:
            with self.subTest(format=format):
                job=self.job('export',format=format)
                response=self.client.get(job['result']['download_url'])
                self.assertEqual(response.status_code,200)
                content=b''.join(response.streaming_content)
                if format in {'csv','tsv'}: self.assertIn(b"'=1+1",content)
                if format=='json':self.assertEqual(json.loads(content)[0]['b'],'=1+1')
                if format=='parquet':self.assertEqual(pd.read_parquet(io.BytesIO(content)).iloc[0]['b'],'=1+1')
                if format=='xlsx':self.assertEqual(pd.read_excel(io.BytesIO(content)).iloc[0]['b'],"'=1+1")

    def test_authentication_owner_isolation_csrf_and_persistence(self):
        imported=self.upload();export=self.job('export',format='csv')
        state=self.state();self.client.logout()
        self.assertIn(self.client.get('/api/state/').status_code,[401,403])
        self.client.force_login(self.other)
        self.assertIsNone(self.state())
        self.assertEqual(self.client.get('/api/jobs/'+imported['id']+'/').status_code,404)
        self.assertEqual(self.client.get(export['result']['download_url']).status_code,404)
        upload_id=Job.objects.get(pk=imported['id']).payload['upload_id']
        self.assertEqual(self.client.post('/api/ingest/',{'upload_id':upload_id},format='json').status_code,404)
        self.client.force_login(self.user);self.assertEqual(self.state()['id'],state['id'])
        csrf=APIClient(enforce_csrf_checks=True);csrf.force_login(self.user)
        self.assertEqual(csrf.post('/api/jobs/',{},format='json').status_code,403)

    def test_datetime_conversion_exports_excel_as_utc(self):
        self.upload(b'date\n2026-01-01T10:00:00+05:30\n')
        self.job('clean', operation={'action':'convert','method':'datetime'})
        result=self.job('export', format='xlsx')
        content=b''.join(self.client.get(result['result']['download_url']).streaming_content)
        self.assertEqual(pd.read_excel(io.BytesIO(content)).iloc[0,0].hour,4)

    def test_excel_two_stage_import_and_oversize_body(self):
        data=io.BytesIO()
        pd.DataFrame({'score':[1,2]}).to_excel(data,index=False,sheet_name='Observations')
        result=self.upload(data.getvalue(),name='book.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        self.assertIsNone(self.state())
        self.assertEqual(result['result']['choose']['names'],['Observations'])
        response=self.client.post('/api/ingest/',{'upload_id':result['result']['upload_id'],
            'options':{'sheets':['Observations']}},format='json')
        self.assertEqual(response.json()['status'],'succeeded')
        self.assertEqual(self.state()['profile']['rows'],2)
        with override_settings(MAX_UPLOAD_BYTES=2):
            response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile('large.csv',b'a\n1\n',content_type='text/csv')},format='multipart')
            self.assertEqual(response.status_code,400)

    def test_concurrent_submission_and_duplicate_task_delivery(self):
        with patch('workspace.api.run_job.delay'):
            first=self.upload()
            response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile('b.csv',b'b\n1\n',content_type='text/csv')},format='multipart')
            self.assertEqual(response.status_code,400)
        from workspace.tasks import run_job
        run_job(first['id']); original=self.state()['id']
        run_job(first['id'])
        self.assertEqual(self.state()['id'],original)
        self.assertEqual(Workspace.objects.get(owner=self.user).dataset_set.count(),1)

    def test_invalid_job_preserves_revision_and_stale_request_is_rejected(self):
        self.upload();state=self.state()
        response=self.client.post('/api/jobs/',{'kind':'clean','dataset_id':state['id'],'revision':0,
            'operation':{'action':'fill','method':'mean'}},format='json')
        self.assertEqual(response.status_code,503)
        self.assertEqual(self.state()['revision'],0)
        self.job('clean',operation={'action':'fill','method':'mean','selection':{'columns':['a']}})
        response=self.client.post('/api/jobs/',{'kind':'undo','dataset_id':state['id'],'revision':0},format='json')
        self.assertEqual(response.status_code,400)

    def test_outliers_require_preview_and_detection_is_revision_scoped(self):
        self.upload(b'a\n1\n2\n2\n3\n2\n100\n')
        state=self.state()
        response=self.client.post('/api/jobs/',{'kind':'clean','dataset_id':state['id'],'revision':0,
            'operation':{'action':'remove_outliers'}},format='json')
        self.assertEqual(response.json()['status'],'failed');self.assertEqual(self.state()['profile']['rows'],6)
        detected=self.job('detect',operation={'action':'detect_outliers','method':'iqr','selection':{'columns':['a']}})
        rows=self.client.get('/api/preview/',{'detection':detected['id']}).json()['rows']
        self.assertEqual(sum(row['flagged'] for row in rows),1)
        self.job('clean',operation={'action':'remove_outliers'},detection_job=detected['id'])
        self.assertEqual(self.state()['profile']['rows'],5)
        self.job('undo');self.assertEqual(self.state()['profile']['rows'],6)

    def test_recipe_failure_is_atomic_and_clear_is_explicit(self):
        self.upload()
        state=self.state()
        recipe={'version':1,'schema':['a','b'],'operations':[
            {'action':'fill','method':'mean','selection':{'columns':['a']}}, {'action':'unknown'}]}
        response=self.client.post('/api/jobs/',{'kind':'replay','dataset_id':state['id'],'revision':0,'recipe':recipe},format='json')
        self.assertEqual(response.json()['status'],'failed');self.assertEqual(self.state()['revision'],0)
        self.job('clear');self.assertIsNone(self.state())

    @override_settings(CELERY_TASK_ALWAYS_EAGER=False)
    def test_broker_failure_does_not_leave_workspace_locked(self):
        with patch('workspace.api.run_job.delay',side_effect=ConnectionError('offline')):
            response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile('a.csv',b'a\n1\n',content_type='text/csv')},format='multipart')
        self.assertEqual(response.status_code,503)
        self.assertFalse(Job.objects.filter(status__in=['queued','running']).exists())

    def test_upload_rate_limit(self):
        with patch('workspace.api.run_job.delay'):
            for index in range(20):
                response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile('a.csv',b'a\n1\n',content_type='text/csv')},format='multipart')
                self.assertEqual(response.status_code,202)
                Job.objects.filter(status='queued').update(status='failed')
            response=self.client.post('/api/ingest/',{'file':SimpleUploadedFile('a.csv',b'a\n1\n',content_type='text/csv')},format='multipart')
            self.assertEqual(response.status_code,429)

    def test_templates_render_and_preview_is_bounded(self):
        response=self.client.get('/')
        self.assertEqual(response.status_code,200)
        self.assertContains(response,'Your data, a little clearer.')
        content='a,b\n'+'\n'.join(f'{i},v{i}' for i in range(155))
        self.upload(content.encode())
        response=self.client.get('/api/preview/',{'page_size':10000,'page':2,'sort':'a','direction':'desc'}).json()
        self.assertEqual(len(response['rows']),55)
        self.assertEqual(response['rows'][0]['values']['a'],54)
        self.assertEqual(response['total'],155)
