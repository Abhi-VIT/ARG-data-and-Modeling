"""Bounded parsers. SQL dumps are interpreted as literal data, never executed."""
import csv
import io
import ipaddress
import json
from pathlib import Path
import re
import socket
import sqlite3
import zipfile
from urllib.parse import urlsplit, parse_qs

import chardet
from defusedxml import ElementTree
from django.conf import settings
import pandas as pd
import regex
from sqlalchemy import create_engine
import sqlglot
from sqlglot import exp
import urllib3

ROW_ID = '__arg_rowid__'
MIMES = {
    '.csv': {'text/csv', 'application/csv', 'text/plain', 'application/vnd.ms-excel'},
    '.tsv': {'text/tab-separated-values', 'text/plain'}, '.txt': {'text/plain'},
    '.json': {'application/json', 'text/json', 'text/plain'},
    '.xml': {'application/xml', 'text/xml', 'text/plain'},
    '.sql': {'application/sql', 'text/plain', 'text/x-sql'},
    '.db': {'application/vnd.sqlite3', 'application/x-sqlite3'},
    '.parquet': {'application/vnd.apache.parquet', 'application/x-parquet'},
    '.xlsx': {'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'},
    '.xls': {'application/vnd.ms-excel'},
}


class DataError(ValueError):
    pass


def validate_file(path, name, mime):
    path = Path(path)
    ext = Path(name).suffix.lower()
    if ext not in MIMES:
        raise DataError('Unsupported format. Choose CSV, TSV, TXT, JSON, Excel, Parquet, XML, SQL or SQLite.')
    if not 0 < path.stat().st_size <= settings.MAX_UPLOAD_BYTES:
        raise DataError(f'File must be nonempty and at most {settings.MAX_UPLOAD_BYTES // 1024**2} MB.')
    mime = mime.split(';')[0].lower().strip()
    if mime not in MIMES[ext] | {'application/octet-stream', ''}:
        raise DataError('The reported file type does not match its extension.')
    with path.open('rb') as stream:
        head = stream.read(8192)
    signatures = {'.xlsx': b'PK\x03\x04', '.xls': b'\xd0\xcf\x11\xe0',
                  '.parquet': b'PAR1', '.db': b'SQLite format 3\x00'}
    if ext in signatures:
        if not head.startswith(signatures[ext]):
            raise DataError('File signature does not match its extension.')
    else:
        if head.startswith((b'MZ', b'PK\x03\x04', b'\x7fELF', b'%PDF', b'SQLite format 3')):
            raise DataError('A binary file cannot be imported as text.')
        if b'\x00' in head and not head.startswith((b'\xff\xfe', b'\xfe\xff')):
            raise DataError('Unexpected binary content in text file.')
    if ext == '.xlsx':
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > settings.MAX_UPLOAD_BYTES * 4:
                raise DataError('The expanded workbook exceeds the size limit.')
            if '[Content_Types].xml' not in archive.namelist() or 'xl/workbook.xml' not in archive.namelist():
                raise DataError('This ZIP is not an XLSX workbook.')
    return ext


def text_content(path, encoding=None):
    data = Path(path).read_bytes()
    detected = chardet.detect(data[:65536])['encoding'] or 'utf-8'
    try:
        return data.decode(encoding or detected)
    except (UnicodeError, LookupError) as exc:
        raise DataError('Cannot decode this file. Choose the correct encoding.') from exc


def normalize_frame(frame):
    if len(frame) > settings.MAX_ROWS or len(frame.columns) > settings.MAX_COLUMNS:
        raise DataError(f'Dataset exceeds {settings.MAX_ROWS:,} rows or {settings.MAX_COLUMNS} columns.')
    if len(frame.columns) == 0:
        raise DataError('The file contains no columns.')
    frame = frame.copy().reset_index(drop=True)
    frame.columns = [str(c) for c in frame.columns]
    if not frame.columns.is_unique or ROW_ID in frame.columns:
        raise DataError(f'Column names must be unique and cannot use reserved name {ROW_ID}.')
    if any(not c.strip() or len(c) > 200 for c in frame.columns):
        raise DataError('Column names must contain 1–200 characters.')
    # Nested cells and mixed Python objects cannot be stored consistently in Parquet.
    for column in frame.select_dtypes(include=['object']).columns:
        frame[column] = frame[column].map(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        types = {type(v) for v in frame[column].dropna()}
        if len(types) > 1 or (types and types <= {str}):
            frame[column] = frame[column].astype('string')
    return frame


def literal(node):
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Boolean):
        return node.this
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and not node.this.is_string:
        return -literal(node.this)
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        return float(node.this) if any(c in node.this.lower() for c in '.e') else int(node.this)
    raise DataError('SQL dumps may contain literal INSERT VALUES only; expressions and queries are rejected.')


def sql_tables(path):
    if Path(path).stat().st_size > min(settings.MAX_UPLOAD_BYTES, 20 * 1024**2):
        raise DataError('SQL dumps are limited to 20 MB. Export larger tables as CSV or Parquet.')
    tables = {}
    try:
        statements = sqlglot.parse(text_content(path), read='mysql')
    except sqlglot.errors.ParseError as exc:
        raise DataError('Unsupported SQL syntax. Use CREATE TABLE and literal INSERT VALUES, or export as CSV.') from exc
    for statement in statements:
        if statement is None:
            continue
        if isinstance(statement, exp.Create) and str(statement.args.get('kind')).upper() == 'TABLE':
            schema = statement.this
            if not isinstance(schema, exp.Schema) or statement.args.get('expression') is not None:
                raise DataError('Only CREATE TABLE with explicit columns is supported.')
            name = schema.this.name
            if name in tables:
                raise DataError('Repeated CREATE TABLE in SQL dump.')
            columns = [c.name for c in schema.expressions if isinstance(c, exp.ColumnDef)]
            if not columns or len(columns) > settings.MAX_COLUMNS or len(tables) >= 100:
                raise DataError('SQL schema exceeds the table/column limit or has no columns.')
            tables[name] = {'columns': columns, 'rows': []}
        elif isinstance(statement, exp.Insert) and isinstance(statement.expression, exp.Values):
            target = statement.this
            name = target.this.name if isinstance(target, exp.Schema) else target.name
            if name not in tables:
                raise DataError('CREATE TABLE must precede its INSERT statements.')
            table = tables[name]
            columns = [c.name for c in target.expressions] if isinstance(target, exp.Schema) else table['columns']
            if not set(columns) <= set(table['columns']) or len(set(columns)) != len(columns):
                raise DataError('INSERT has unknown or duplicate columns.')
            for row in statement.expression.expressions:
                if not isinstance(row, exp.Tuple) or len(row.expressions) != len(columns):
                    raise DataError('INSERT row does not match its columns.')
                table['rows'].append(dict(zip(columns, map(literal, row.expressions))))
                if len(table['rows']) > settings.MAX_ROWS:
                    raise DataError('SQL table exceeds the row limit.')
        else:
            raise DataError('SQL dump rejected: only CREATE TABLE and literal INSERT VALUES are allowed. No SQL is executed.')
    if not tables:
        raise DataError('No supported tables found in SQL dump.')
    return tables


def sqlite_connection(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    connection.enable_load_extension(False)
    connection.execute('PRAGMA trusted_schema=OFF')
    connection.execute('PRAGMA query_only=ON')
    return connection


def choices(path, ext):
    if ext in {'.xls', '.xlsx'}:
        with pd.ExcelFile(path) as book:
            return {'kind': 'sheets', 'names': book.sheet_names}
    if ext == '.db':
        connection = sqlite_connection(path)
        try:
            names = [r[0] for r in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "AND upper(sql) NOT LIKE '%VIRTUAL%' LIMIT 101")]
            if not names or len(names) > 100:
                raise DataError('SQLite file must contain 1–100 ordinary tables.')
            return {'kind': 'tables', 'names': names}
        finally:
            connection.close()
    if ext == '.sql':
        return {'kind': 'tables', 'names': list(sql_tables(path))}
    return None


def parse_file(path, name, mime, options):
    ext = validate_file(path, name, mime)
    if ext in {'.csv', '.tsv', '.txt'}:
        content = text_content(path, options.get('encoding') or None)
        delimiter = options.get('delimiter') or ('\t' if ext == '.tsv' else None)
        if delimiter == '\\t':
            delimiter = '\t'
        header = 0 if options.get('header', True) else None
        if options.get('regex_delimiter'):
            if not delimiter or len(delimiter) > 100:
                raise DataError('Provide a delimiter regex with 1–100 characters.')
            pattern = regex.compile(delimiter)
            rows = []
            for line in io.StringIO(content):
                rows.append(pattern.split(line.rstrip('\r\n'), timeout=0.05))
                if len(rows) > settings.MAX_ROWS + 1:
                    raise DataError('Text file exceeds the row limit.')
            frame = pd.DataFrame(rows[1:], columns=rows[0]) if header == 0 else pd.DataFrame(rows)
            frame = frame.replace('', pd.NA)
            for col in frame:
                try:
                    frame[col] = pd.to_numeric(frame[col])
                except (TypeError, ValueError):
                    pass
        else:
            if delimiter is None:
                try:
                    delimiter = csv.Sniffer().sniff(content[:32768], delimiters=',;\t| ').delimiter
                except csv.Error:
                    delimiter = ','
            if len(delimiter) != 1:
                raise DataError('Use one delimiter character or enable regex delimiter.')
            frame = pd.read_csv(io.StringIO(content), sep=delimiter, header=header, nrows=settings.MAX_ROWS + 1)
    elif ext == '.json':
        data = json.loads(text_content(path))
        depth = int(options.get('depth', 3))
        if not 0 <= depth <= 20:
            raise DataError('Nesting depth must be between 0 and 20.')
        if isinstance(data, dict):
            record_key = options.get('record_key')
            data = data[record_key] if record_key else [data]
        if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
            raise DataError('JSON must contain records (objects). Set the record key for a wrapped list.')
        if len(data) > settings.MAX_ROWS:
            raise DataError('JSON exceeds the row limit.')
        frame = pd.json_normalize(data, max_level=depth)
    elif ext in {'.xlsx', '.xls'}:
        selected = options.get('sheets', [])
        if not selected or not isinstance(selected, list):
            raise DataError('Choose one or more worksheets.')
        frames = pd.read_excel(path, sheet_name=selected, nrows=settings.MAX_ROWS + 1)
        if len(frames) > 1:
            if any('_source_sheet' in f.columns for f in frames.values()):
                raise DataError('Rename reserved _source_sheet column before combining sheets.')
            frame = pd.concat([f.assign(_source_sheet=s) for s, f in frames.items()], ignore_index=True)
        else:
            frame = next(iter(frames.values()))
    elif ext == '.parquet':
        import pyarrow.parquet as pq
        metadata = pq.ParquetFile(path).metadata
        if metadata.num_rows > settings.MAX_ROWS or metadata.num_columns > settings.MAX_COLUMNS:
            raise DataError('Parquet exceeds row/column limits.')
        frame = pd.read_parquet(path)
    elif ext in {'.db', '.sql'}:
        table = options.get('table')
        if ext == '.sql':
            tables = sql_tables(path)
            if table not in tables:
                raise DataError('Choose an available SQL table.')
            frame = pd.DataFrame(tables[table]['rows'], columns=tables[table]['columns'])
        else:
            if table not in choices(path, ext)['names']:
                raise DataError('Choose an available SQLite table.')
            engine = create_engine('sqlite://', creator=lambda: sqlite_connection(path))
            try:
                quoted = '"' + table.replace('"', '""') + '"'
                frame = pd.read_sql_query(f'SELECT * FROM {quoted} LIMIT {settings.MAX_ROWS + 1}', engine)
            finally:
                engine.dispose()
    elif ext == '.xml':
        content = text_content(path)
        ElementTree.fromstring(content)  # Reject entities/DTD expansion before pandas sees it.
        xpath = options.get('xpath', './*') or './*'
        if len(xpath) > 200 or not re.fullmatch(r'[\w./*\-]+', xpath):
            raise DataError('XPath may contain element names, /, ., * and hyphens only.')
        frame = pd.read_xml(io.StringIO(content), xpath=xpath, parser='etree')
    else:
        raise DataError('Unsupported file.')
    return normalize_frame(frame)


def fetch_csv(url, destination):
    """Allowlisted HTTPS only; pin the validated IP to prevent DNS rebinding."""
    parts = urlsplit(url)
    host = parts.hostname or ''
    if parts.scheme != 'https' or host not in settings.IMPORT_URL_HOSTS or parts.username or parts.password or parts.port not in (None, 443):
        raise DataError('Use an HTTPS URL on an administrator-approved import host. You can also upload the file.')
    if parts.fragment:
        raise DataError('Remove the URL fragment. Use gid in the query for a Google worksheet.')
    path = parts.path + ('?' + parts.query if parts.query else '')
    if host == 'docs.google.com':
        match = re.fullmatch(r'/spreadsheets/d/([\w-]+)(?:/.*)?', parts.path)
        if not match:
            raise DataError('Provide a public Google Sheets spreadsheet URL.')
        gid = parse_qs(parts.query).get('gid', ['0'])[0]
        if not gid.isdigit():
            raise DataError('Google sheet gid must be numeric.')
        path = f'/spreadsheets/d/{match[1]}/gviz/tq?tqx=out:csv&gid={gid}'
    addresses = {item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise DataError('Import URLs cannot resolve to private or reserved networks.')
    pool = urllib3.HTTPSConnectionPool(sorted(addresses)[0], port=443, server_hostname=host,
                                      assert_hostname=host, timeout=urllib3.Timeout(connect=5, read=20))
    try:
        response = pool.request('GET', path, headers={'Host': host, 'Accept': 'text/csv', 'Accept-Encoding': 'identity'},
                                redirect=False, retries=False, preload_content=False)
        try:
            if response.status != 200:
                raise DataError('URL must return the CSV directly with HTTP 200. Redirects/private sheets are not supported.')
            mime = response.headers.get('Content-Type', '').split(';')[0].lower()
            if mime not in {'text/csv', 'text/plain', 'application/csv', 'text/tab-separated-values'}:
                raise DataError('URL did not return a supported CSV content type.')
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise DataError('Compressed URL responses are not supported. Upload the downloaded CSV.')
            if int(response.headers.get('Content-Length', '0')) > settings.MAX_UPLOAD_BYTES:
                raise DataError('Remote file exceeds the upload limit.')
            size = 0
            with Path(destination).open('wb') as output:
                for chunk in response.stream(65536, decode_content=False):
                    size += len(chunk)
                    if size > settings.MAX_UPLOAD_BYTES:
                        raise DataError('Remote file exceeds the upload limit.')
                    output.write(chunk)
            return mime
        finally:
            response.release_conn()
    finally:
        pool.close()
