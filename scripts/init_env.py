"""Create local configuration without overwriting an existing .env or printing secrets."""
from pathlib import Path
import secrets
import sys

root = Path(__file__).resolve().parent.parent
if sys.version_info[:3] != (3, 12, 10):
    raise SystemExit('This project pins Python 3.12.10. Install that version before creating the venv.')
if sys.prefix == sys.base_prefix:
    raise SystemExit('Activate the project venv before running this script.')
destination = root / '.env'
if destination.exists():
    print('.env already exists; no changes made.')
else:
    content = (root / '.env.example').read_text(encoding='utf-8')
    content = content.replace('SECRET_KEY=\n', 'SECRET_KEY=' + secrets.token_urlsafe(50) + '\n', 1)
    destination.write_text(content, encoding='utf-8')
    print('Created ignored .env with a random secret. Configure DATABASE_URL and REDIS_URL as needed.')
