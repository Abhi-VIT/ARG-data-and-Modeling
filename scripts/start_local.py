"""Start the persistent, loopback-only app and its embedded Celery worker."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = ROOT / 'media' / 'local'
INSTANCE_ID = hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:24]


def ensure_environment():
    expected = (ROOT / '.python-version').read_text().strip()
    actual = '.'.join(map(str, sys.version_info[:3]))
    if actual != expected or sys.prefix == sys.base_prefix:
        raise RuntimeError(f'Use the project venv with Python {expected}. Current Python: {actual}.')
    missing = []
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, version = line.split('==', 1)
        name = name.split('[')[0]
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        # PyTorch CPU/CUDA wheels append a local build tag to the pinned release.
        if (installed.split('+')[0] if name == 'torch' and installed else installed) != version:
            missing.append(line)
    if missing:
        print('Installing required dependencies into venv (internet needed on first setup)...', flush=True)
        subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'], check=True, cwd=ROOT)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'init_env.py')], check=True, cwd=ROOT)
    if not (ROOT / 'static' / 'css' / 'app.css').is_file():
        raise RuntimeError('Frontend assets are missing. Run npm ci and npm run build in the project folder.')


def health(port):
    try:
        with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}/_local/health/', timeout=1) as response:
            return json.load(response).get('instance') == INSTANCE_ID
    except (OSError, ValueError):
        return False


def open_browser(port, enabled):
    url = f'http://127.0.0.1:{port}/'
    print(f'Open: {url}', flush=True)
    if enabled:
        try:
            if not webbrowser.open(url):
                print('Open the URL above in your browser.', flush=True)
        except OSError:
            print('Could not open the browser automatically. Use the URL above.', flush=True)


def acquire_lock(port):
    """OS-released lock prevents two workers from writing the same local database."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL_DIR / 'launcher.lock').open('a+b')
    if lock.seek(0, 2) == 0:
        lock.write(b' ')
        lock.flush()
    lock.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.seek(1)
        try:
            running_port = json.loads(lock.read().decode())['port']
        except (ValueError, KeyError):
            running_port = port
        lock.close()
        return None, running_port
    lock.seek(1)
    lock.truncate()
    lock.write(json.dumps({'port': port}).encode())
    lock.flush()
    return lock, port


def serve(port, browser_enabled):
    import django
    django.setup()
    from celery.worker.worker import WorkController
    from django.contrib.staticfiles.handlers import StaticFilesHandler
    from django.core.management import call_command
    from django.core.servers.basehttp import ThreadedWSGIServer, WSGIRequestHandler
    from django.core.wsgi import get_wsgi_application
    from django.utils import timezone
    from project.celery import app
    from workspace.models import Job
    from workspace.tasks import run_job, run_deep_job

    # Bind before changing any job state. Never open another application's occupied port.
    try:
        server = ThreadedWSGIServer(('127.0.0.1', port), WSGIRequestHandler, allow_reuse_address=False)
    except OSError as exc:
        raise RuntimeError(f'Port {port} is already in use. Close the other server, or run start.bat --port 8001.') from exc
    worker = None
    worker_thread = None
    server_thread = None
    try:
        call_command('check')
        call_command('migrate', verbosity=1)
        Job.objects.filter(status='running').update(
            status='failed', updated_at=timezone.now(),
            message='App closed during this job. Review the current dataset revision before retrying.')

        ready = threading.Event()
        app.loader.import_default_modules()
        app.finalize()
        app.set_current()
        app.set_default()
        worker = WorkController(app=app, pool='solo', concurrency=1, hostname='arg-local@localhost',
                                queues=['celery','deep'],
                                ready_callback=lambda consumer: ready.set(),
                                without_heartbeat=True, without_mingle=True, without_gossip=True)
        worker_thread = threading.Thread(target=worker.start, name='arg-worker', daemon=True)
        worker_thread.start()
        deadline = time.monotonic() + 20
        while not ready.wait(.2):
            if not worker_thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError('The background worker did not start. See the console output above.')

        # Queued jobs are durable in SQLite even though the local broker is in memory.
        for job_id,kind in Job.objects.filter(status='queued').values_list('id','kind'):
            if kind in {'deep','image_ingest'}:run_deep_job.apply_async(args=[str(job_id)],queue='deep')
            else:run_job.delay(str(job_id))

        application = StaticFilesHandler(get_wsgi_application())

        def local_application(environ, start_response):
            if environ.get('PATH_INFO') == '/_local/health/':
                body = json.dumps({'instance': INSTANCE_ID}).encode()
                start_response('200 OK', [('Content-Type', 'application/json'), ('Content-Length', str(len(body)))])
                return [body]
            return application(environ, start_response)

        server.set_app(local_application)
        server_thread = threading.Thread(target=server.serve_forever, name='arg-web', daemon=True)
        server_thread.start()
        if not health(port):
            raise RuntimeError('The local web server did not become ready.')
        print('\nARG Data Studio is ready. Create an account on your first visit.', flush=True)
        print(f'Your data is saved in: {LOCAL_DIR}', flush=True)
        print('Keep this window open. Press Ctrl+C to stop the app.\n', flush=True)
        open_browser(port, browser_enabled)
        while server_thread.is_alive() and worker_thread.is_alive():
            time.sleep(.5)
        raise RuntimeError('The web server or background worker stopped unexpectedly. Restart start.bat.')
    except KeyboardInterrupt:
        print('\nStopping ARG Data Studio...', flush=True)
    finally:
        if server_thread:
            server.shutdown()
            server_thread.join(timeout=5)
        server.server_close()
        if worker:
            worker.stop()
        if worker_thread:
            worker_thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true', help='Start without opening a browser.')
    parser.add_argument('--check', action='store_true', help='Check setup without starting services.')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use a port between 1024 and 65535.')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if args.check:
        ensure_environment()
        print('Local launcher setup is ready.', flush=True)
        return
    lock, running_port = acquire_lock(args.port)
    if lock is None:
        if health(running_port):
            print('ARG Data Studio is already running.', flush=True)
            open_browser(running_port, not args.no_browser)
            return
        raise RuntimeError('ARG is starting or stopping in another window. Wait for that window, then try again.')
    try:
        ensure_environment()
        os.environ['DJANGO_SETTINGS_MODULE'] = 'project.local_settings'
        serve(args.port, not args.no_browser)
    finally:
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'\nStartup failed: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
