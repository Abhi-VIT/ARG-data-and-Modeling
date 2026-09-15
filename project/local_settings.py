"""Single-computer launcher settings; separate from the test and Docker databases."""
from .settings import *  # noqa: F403

DEBUG = True
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']
LOCAL_DIR = BASE_DIR / 'media' / 'local'  # noqa: F405
DATABASES = {'default': {
    'ENGINE': 'django.db.backends.sqlite3',
    'NAME': LOCAL_DIR / 'metadata.sqlite3',
    'OPTIONS': {'timeout': 30},
}}
MEDIA_ROOT = LOCAL_DIR / 'datasets'
CELERY_BROKER_URL = 'memory://'
CELERY_RESULT_BACKEND = 'cache+memory://'
CELERY_TASK_ALWAYS_EAGER = False
CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SESSION_COOKIE_NAME = 'arg_local_session'
CSRF_COOKIE_NAME = 'arg_local_csrf'
# Keep Django's normal password hashing and validators from the base settings.
