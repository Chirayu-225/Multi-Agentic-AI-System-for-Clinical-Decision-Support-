"""
Run the test suite against a real PostgreSQL server, without Docker.

    pip install pgserver psycopg2-binary
    python scripts/run_tests_on_postgres.py                    # the whole suite
    python scripts/run_tests_on_postgres.py tests/test_job_queue.py -q   # or anything else pytest accepts

pgserver ships a private PostgreSQL that lives in a temp folder and is removed afterwards. The
suite is pointed at a throw-away database called medsight_test (the test setup refuses any
database whose name does not end in "_test", because the tests drop all tables between cases).
This is how the FOR UPDATE SKIP LOCKED claim path gets exercised, which SQLite cannot do.
"""
import os
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit

try:
    import pgserver
    import psycopg2  # noqa: F401  (the driver SQLAlchemy uses for postgresql:// URLs)
except ImportError as e:
    sys.exit(f"Missing package: {e.name}. Run: pip install pgserver psycopg2-binary")


def test_database_url(server_uri: str, name: str = "medsight_test") -> str:
    """Point a pgserver URI at `name`, whatever shape it has: TCP ("...:5432/postgres") on Windows,
    or a socket URI with a query string ("...@/postgres?host=/tmp/x") on Linux."""
    parts = urlsplit(server_uri)
    return urlunsplit(parts._replace(path=f"/{name}"))

DEFAULT_TESTS = ["tests", "-q"]       # the whole suite; pass your own arguments to narrow it

backend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
workdir = tempfile.mkdtemp(prefix="medsight-pg-")
server = pgserver.get_server(workdir)
try:
    server.psql("CREATE DATABASE medsight_test;")
    url = test_database_url(server.get_uri())
    assert urlsplit(url).path == "/medsight_test", url
    env = {**os.environ, "MEDSIGHT_TEST_DATABASE_URL": url}
    args = sys.argv[1:] or DEFAULT_TESTS
    code = subprocess.call([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=backend, env=env)
finally:
    server.cleanup()
sys.exit(code)
