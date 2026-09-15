import os
os.environ.setdefault('SECRET_KEY', 'test-only-key-not-for-deployment')
os.environ.setdefault('DATABASE_URL', 'sqlite://:memory:')
os.environ.setdefault('DEBUG', 'True')
from .settings import *  # noqa: F403

CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_BROKER_URL = 'memory://'
CELERY_RESULT_BACKEND = 'cache+memory://'
CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']
