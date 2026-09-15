"""Loopback-only UI test harness: SQLite + a real Celery worker on an in-memory broker.

Not a deployment entry point. Runtime data lives under ignored .tmp/ui-smoke.
Use the regular project settings and Redis/PostgreSQL for normal operation.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
runtime = ROOT / '.tmp' / 'ui-smoke'
runtime.mkdir(parents=True, exist_ok=True)
os.environ['DJANGO_SETTINGS_MODULE'] = 'project.test_settings'
os.environ['DATABASE_URL'] = 'sqlite:///' + (runtime / 'metadata.sqlite3').as_posix()
os.environ['MEDIA_ROOT'] = str(runtime / 'media')
os.environ['DEBUG'] = 'True'
os.environ['ALLOWED_HOSTS'] = '127.0.0.1,localhost'
import django
django.setup()
from django.conf import settings
settings.CELERY_TASK_ALWAYS_EAGER = False
from django.core.management import call_command
call_command('migrate', verbosity=0)
from django.core.servers.basehttp import run
from django.contrib.staticfiles.handlers import StaticFilesHandler
from django.core.wsgi import get_wsgi_application
from celery.contrib.testing.worker import start_worker
from project.celery import app

print('UI test harness: http://127.0.0.1:8787 (isolated test database; register a test account)', flush=True)
with start_worker(app, pool='solo', concurrency=1, perform_ping_check=False, loglevel='WARNING'):
    run('127.0.0.1', 8787, StaticFilesHandler(get_wsgi_application()), threading=True)
