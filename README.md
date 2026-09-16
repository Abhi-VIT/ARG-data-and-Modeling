# ARG Data Studio

A Django workspace for preparing, analyzing, and modeling data without writing Python. **Phases 1–4 are implemented:** ingestion, cleaning, export, statistics, classical machine learning, PyTorch deep learning, and interactive plots.

**Import → inspect → prepare → understand → train → export.**

![Animated tour of the real data explorer, statistical reports, classical models and neural training interface](docs/media/platform-tour.gif)

*A guided tour of the actual application. All screenshots use the synthetic water-quality sample or generated test images in an isolated demo workspace. Results illustrate the interface, not model performance on a real research dataset. GIFs are looping sequences of captured UI states; pauses are chosen for readability and are not execution-time benchmarks.*

## Contents

- [Project purpose and workflow](#project-purpose-and-workflow)
- [Implemented capabilities](#implemented)
- [Local setup and Windows launcher](#local-setup)
- [Workspace guide and UI map](#workspace-guide)
- [Import formats and parsing reference](#import-formats-and-parsing-reference)
- [Cleaning operations reference](#cleaning-operations-reference)
- [History, recipes and exports](#history-recipes-and-exports)
- [Phase 2: statistics and diagnostics](#phase-2-analysis-guide)
- [Phase 3: classical machine learning](#phase-3-modeling-guide)
- [Phase 4: deep learning](#phase-4-deep-learning-guide)
- [Limits and behavior](#limits-and-behavior)
- [Architecture, data model and job lifecycle](#architecture)
- [Configuration reference](#configuration-reference)
- [API and request examples](#api)
- [Source code map](#source-code-map)
- [Validation and contributing](#validation)
- [Operations, backups and troubleshooting](#operations)
- [FAQ and current boundaries](#faq-and-current-boundaries)
- [Documentation media](#documentation-media)

## Project purpose and workflow

ARG Data Studio brings data preparation and modeling into one authenticated browser workspace. A user can inspect a dataset, apply explicit transformations, examine statistics, train a model, and download the resulting data or artifacts. The original research files remain in this repository, but the running application is the Django project under `project/` and `workspace/`.

The platform is useful for exploratory work and reproducible preparation of datasets that fit in one worker's memory. It does not require the user to write Python for the supported browser workflows. Scientific choices still belong to the user: an available test or estimator is not automatically appropriate for a particular study.

### A typical session

1. Start the app, create an account, and import a file or choose **Try sample data**.
2. Inspect row/column counts, data types, missingness and duplicates before changing data.
3. Select the relevant cells, rows or columns. Apply a cleaning operation and inspect its new revision.
4. Save a preparation recipe if those same steps will be used again.
5. Use Statistics to examine associations, fit an explanatory regression, compare groups or draw a plot.
6. Use Models for predictive regression/classification, clustering or dimensionality reduction.
7. Use Deep learning for configurable neural networks, sequence prediction or image classification.
8. Export cleaned data, JSON reports, chart images, fitted model bundles, checkpoints or row-level outputs.

### Concepts used throughout this guide

| Term | Meaning in this project |
| --- | --- |
| Workspace | One private workspace per Django user; it has one active tabular dataset at a time. |
| Dataset | An imported table with a name, original column schema and revision cursor. |
| Revision | A saved Parquet snapshot after import or a successful cleaning step. Revision 0 is the import. |
| Job | A persistent record of work: queued, running, succeeded or failed. The UI polls this record. |
| Report | A private JSON artifact containing tables, figures, metrics, parameters and source metadata. |
| Feature | An input column supplied to a statistical or machine-learning model. |
| Target | The outcome to predict. Classification targets identify classes; regression targets are numeric. |
| Holdout | Rows excluded from fitting, used later to evaluate predictions. Neural training also reserves validation rows. |
| Artifact | A file produced by a job, such as cleaned data, a report, `.pkl` model, `.pt` checkpoint or output CSV. |

### Technology stack

| Layer | Components and purpose |
| --- | --- |
| Web application | Python 3.12.10, Django 5.2.17, Django REST Framework 3.16.1; templates, authentication and APIs. |
| Browser UI | Alpine.js 3.14.9 for reactive state, HTMX 2.0.4 for workspace status, Tailwind CSS 3.4.17 for styling. |
| Charts | Plotly Python 6.3.0 builds figure JSON; Plotly.js 3.1.0 renders interactive charts and PNG exports. |
| Background work | Celery 5.5.3; Redis is the hosted broker/cache. The Windows launcher embeds a real worker using an in-memory broker. |
| Metadata | PostgreSQL in the service/Compose setup; SQLite in local and test modes. |
| Table storage and preview | Pandas 2.3.2, NumPy 2.2.6, PyArrow 21.0.0 for private Parquet snapshots; DuckDB 1.4.0 for bounded preview queries. |
| Parsing | chardet, openpyxl, xlrd, SQLAlchemy, sqlglot, defusedxml, lxml and bounded regex handling. |
| Statistics | SciPy 1.16.2, statsmodels 0.14.5 and Pingouin 0.5.5. |
| Classical ML | scikit-learn 1.7.2, XGBoost 3.0.5, LightGBM 4.6.0, UMAP 0.5.9.post2 and joblib 1.5.2. |
| Deep learning | PyTorch 2.8.0 and Pillow 12.3.0; CUDA selection happens in the deep worker. |

Exact Python pins live in [requirements.txt](requirements.txt); browser pins live in [package.json](package.json) and [package-lock.json](package-lock.json). Built frontend assets are included, so normal startup does not download JavaScript from a CDN.

## Implemented

- Django authentication and private, persistent workspaces; every API/job/download checks ownership.
- CSV/TSV/TXT with delimiter/encoding detection, manual overrides, headers, and bounded regex splitting; nested JSON with depth and record-key controls; XLSX/XLS sheet selection; Parquet; XML XPath; SQLite table selection; restricted SQL dumps; public Google Sheets and allowlisted HTTPS CSV URLs.
- Paginated data table, search, column filters/sort, dtype/missingness/unique counts, Plotly missingness heatmap, and numeric header histograms.
- Cell, row, column, range, and whole-dataset selection; cell editing; missing-value filling/dropping; duplicate detection/removal; outlier preview before removal; type/date conversions; text/regex cleaning; categorical encoding; scaling; column drop/rename/split/merge.
- Parquet snapshots with serialized operation history, undo/redo, schema-checked JSON recipe replay, and CSV/TSV/JSON/Parquet/XLSX exports.
- Descriptive summaries; Pearson, Spearman, Kendall, point-biserial, Cramér’s V, and Phi correlations with method suggestions, matrices, heatmaps, pair counts, and p-values.
- Simple, multiple, and custom OLS regression with interactions/polynomials, stepwise selection, coefficient inference, fit statistics, residual tests, VIF, and five influence/diagnostic plots.
- One-way, two-way, Welch, and repeated-measures ANOVA; Tukey HSD for ordinary independent-groups ANOVA; eight EDA plot types. Plotly charts support PNG downloads.
- 28 classical ML estimators: regression, classification, clustering, and dimensionality reduction; configurable train/test splitting; training-only preprocessing; grid/random search; held-out metrics and permutation importance; a live binary classification cutoff; private model downloads and new-file prediction.
- Configurable MLP/ANN, CNN, RNN, LSTM, GRU, and autoencoders; separate image ZIP/folder ingestion; train/validation/test splits; early stopping; live epoch curves; private PyTorch checkpoints and test-output CSVs. A dedicated Celery queue supports CUDA with CPU fallback.
- Celery jobs for all ingestion, cleaning, detection, recipe replay, exports, and statistical analyses. The UI polls persistent job records and resumes polling after reload. Failed jobs preserve the active revision. Reports are private JSON artifacts tied to their original dataset revision.

## Local Setup

### Windows: double-click start.bat

Double-click **`start.bat`** in the project folder. It uses the ignored `venv/`, installs any missing pinned dependencies, checks setup, applies migrations, starts Django and a background Celery worker, and opens [the app](http://127.0.0.1:8000) in your default browser. No Docker, PostgreSQL, or Redis installation is needed for this single-computer mode.

Keep the launcher window open while working. Press **Ctrl+C** to stop. Opening `start.bat` again while it is running reopens the existing app. Startup failures remain visible in the console.

After updating from an earlier phase, stop the running launcher and double-click `start.bat` again. It installs missing packages inside the ignored venv and loads all four phases. Phase 4 adds PyTorch and Pillow; the first installation is a larger download. The embedded worker consumes both ordinary and deep-learning jobs. Compatible CPU/CUDA build tags on the pinned PyTorch version are preserved. UMAP's first run may take longer while its numerical kernels compile.

Your accounts, history, and datasets persist under **`media/local/`** (ignored by Git), separately from the earlier test preview on port 8787 and from the Docker database. Create an account on first use. This mode retains normal Django password hashing and uses a local SQLite database, an in-process Celery worker, and an in-memory broker/cache. Queued jobs are recovered on restart; interrupted running jobs are marked failed so you can inspect the saved revision before retrying. Rate-limit counters reset on restart and native Windows solo workers have no process-based time limits. Use the PostgreSQL/Redis setup below for hosted or multi-user operation.

Optional terminal commands:

```bat
start.bat --port 8001
start.bat --check
start.bat --no-browser
```

### 1. Python and isolated environment

Use **Python 3.12.10**, pinned in `.python-version`. The project venv was created and verified with this version. Install dependencies inside the venv.

```bash
# Verify this reports Python 3.12.10
python3 --version
python3 -m venv venv

# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd.exe):
venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
python scripts/init_env.py
```

On Windows with the Python launcher, use `py -3.12 -m venv venv` after checking `py -3.12 --version`. If activation is unavailable, invoke `venv\Scripts\python.exe` and `venv\Scripts\celery.exe` directly.

`scripts/init_env.py` creates `.env` with a random secret without printing it or overwriting existing settings. Configure the variables documented in `.env.example`. The default metadata database is PostgreSQL. Django refuses to start without `SECRET_KEY`. `venv/`, `.env`, private uploads, databases, and caches are ignored by Git.

### 2. Start PostgreSQL and Redis

Use Docker Desktop with its Linux engine, or existing PostgreSQL/Redis services:

```bash
docker compose up -d db redis
```

The local credentials in `compose.yaml` match `.env.example`. Database and Redis ports bind to loopback only. This service-based setup needs Redis starting in **Phase 1**, including uploads; `start.bat` uses the separate single-computer mode described above.

### 3. Apply migrations and start Django

```bash
python manage.py migrate
python manage.py runserver
```

In another activated terminal:

```bash
celery -A project worker -l info -Q celery --concurrency=2
# Separate terminal for neural training and image validation:
celery -A project worker -l info -Q deep --pool=solo --concurrency=1
```

Use Linux, WSL2, or the Compose worker for normal Celery operation. Celery does not officially support native Windows. For a small **local check only**, a Windows worker can use `celery -A project worker -l info --pool=solo`; process-based task time limits do not apply in that mode. [Celery platform documentation](https://docs.celeryq.dev/en/stable/getting-started/introduction.html).

Open [the app](http://127.0.0.1:8000), create an account, and choose **Try sample data** or **Import dataset**. The sample is synthetic water-quality data. Your dataset persists across sign-ins until replacement or clearing.

### Complete Docker development stack

After creating `.env`:

```bash
docker compose up --build
```

This starts Django, separate ordinary/deep Celery workers, PostgreSQL, and Redis. Web and workers share private media storage. The web service applies migrations before accepting requests. This development stack uses Django's development server. CUDA workers require compatible NVIDIA drivers, a CUDA-enabled PyTorch build, and GPU access; see Phase 4 below.

### Frontend development

Compiled Tailwind CSS and pinned Alpine, HTMX, and Plotly assets are checked in. Normal Python startup needs neither Node nor a CDN. After editing templates/styles:

```bash
npm ci
npm run build
```

Alpine manages selection, forms, and job polling; HTMX refreshes the server-rendered workspace status. The UI uses Django templates, with no separate SPA.

### Choose the correct runtime mode

| Mode | Entry point | Metadata / files | Worker and broker | Intended use |
| --- | --- | --- | --- | --- |
| Persistent Windows local | `start.bat` | `media/local/metadata.sqlite3`; files in `media/local/datasets/` | Embedded solo worker, `celery` + `deep` queues, memory broker | One computer; simplest start. |
| Service development | `manage.py runserver` plus worker commands | `DATABASE_URL` and `MEDIA_ROOT` | Redis and separately started workers | Explicit service setup and development. |
| Compose | `docker compose up --build` | PostgreSQL named volume and shared private-media volume | Redis, ordinary worker, dedicated deep worker | Reproducible development services. |
| Browser smoke preview | `python scripts/smoke_server.py` | `.tmp/ui-smoke/metadata.sqlite3` and `.tmp/ui-smoke/media/` | Real solo worker and memory broker | Isolated UI verification on port 8787. |
| Automated tests | `manage.py test ... --settings=project.test_settings` | Isolated SQLite and temporary media | Eager tasks except explicit enqueue tests | Repeatable automated checks. |

These stores are separate. An account created in the port-8787 demo does not automatically exist in the normal local app. Changing ports with `start.bat --port` does not create a new local workspace database.

### What the Windows launcher does

`start.bat` changes to its own directory, creates `venv/` if necessary and invokes `scripts/start_local.py`. The Python launcher validates the exact Python version and virtual environment, checks pinned packages, installs missing requirements, creates `.env` only if absent, checks built CSS, applies migrations and starts the app on loopback. A filesystem lock prevents a second launcher from writing the same local database. A project-specific health response distinguishes this app from another service occupying the port.

On restart, durable queued jobs are submitted again to the correct queue. Interrupted running jobs are marked failed; they are not silently rerun. For neural jobs, any already-saved checkpoint remains on disk. `--check` verifies environment setup only; it does not start services or run the automated suite. Use ports 1024–65535. Closing the launcher stops its web server and worker.

## Workspace guide

![Data explorer showing sidebar navigation, profile cards, editable data preview, and cleaning controls](docs/media/01-data-explorer.png)

### UI map

| Area | What it shows | How to use it |
| --- | --- | --- |
| Sidebar / compact navigation | Data explorer, Operation history, Statistics, Machine learning, Deep learning | Switch workflow without exporting and reimporting data. On narrower screens, use the content tabs and Deep learning button. |
| Header | Active area, phase, private-workspace/job status | The small status badge refreshes periodically; the main job progress panel polls more frequently. |
| Global actions | Import dataset, Export data, Deep learning | Import replaces the active association only after successful parsing; CNN collections have their own upload route. |
| Profile cards | Rows, columns, types, missing cells, duplicates and estimated frame memory | Values describe the full active revision, not just the visible page or search results. |
| Data table | Rows, column types, missing percentages, unique counts | Scroll wide tables horizontally; choose 25/50/100 rows per page; sort and filter for inspection. |
| Selection summary | Current cell/column/row/range scope | Read it before applying an operation. Clear selection returns to the entire dataset. |
| Cleaning card | Operation-specific controls | Available methods change when the operation changes. Structural transformations require whole columns. |
| Missingness heatmap | Column-by-row-group missing proportions | An overview of where gaps cluster, using at most 40 row groups. Numeric headers also expose 16-bin histograms. |
| Job panel | Message, progress bar, queued/running state | Work continues on the server while the page is closed. Reopen to resume polling. |
| Reports panel | Saved report selector, metrics, plots, tables, notes, downloads | A report retains its source revision. A warning appears if it describes a different/replaced revision. |

### Selection and editing rules

- **Cell:** click a cell to open its editor. Enter saves; Escape cancels. The side editor can explicitly set a missing value.
- **Column:** use the header checkbox for multiple columns, or click the column name for one column.
- **Row:** click its row-number button; multiple rows can be selected.
- **Range:** Shift-click from one selected cell to another for a rectangle, or use the row-range fields. Displayed row numbers start at 1; API row positions start at 0.
- **Whole table:** no explicit column/row selection means all columns/all rows, except Drop selected columns, which requires an explicit column list.
- **Sorting and filtering:** these change the preview only. They do not reorder the stored dataset, redefine a recipe's row positions, or limit later model fitting.
- **Long text:** preview values longer than 2,000 characters are truncated and cannot be edited inline. Exports preserve their full contents.

### Step-by-step workspace actions

1. **Import:** choose a file or URL and parsing options. Excel and SQL/SQLite show available sheets/tables before import. Combining sheets unions columns and adds `_source_sheet`.
2. **Explore:** search all columns, apply contains-filters, and click ↕ to sort. Hover numeric headers for histograms. Only one page of rows is mounted; wide datasets scroll horizontally.
3. **Select:** click a cell for its editor, a column name/checkbox for a column, or a row number for a row. Shift-click a second cell for a rectangular range. Row numbers identify positions in the current revision, even after sorting/filtering. A range uses these numbers, not filtered display order. No selection means the whole dataset.
4. **Clean:** choose an operation and method. To drop columns, explicitly select their header checkboxes, choose **Drop selected columns**, and apply. Undo restores them, and recipes replay the drop. Keep at least one column. Structural column changes require all rows. Numeric methods reject incompatible columns with a clear message. Outlier detection flags rows in amber; review before choosing **Remove flagged rows**.
5. **History:** undo/redo or download a recipe. An edit after undo replaces the redo branch. Replay requires exactly matching ordered input column names; each recorded operation creates its own revision.
6. **Export:** a worker prepares the current revision and starts its download. Reopen Export data to download again.
7. **Statistics:** choose an analysis, explicitly check columns or **Select all applicable**, assign roles, then **Run analysis**. Numeric analyses list excluded nonnumeric columns. Reports retain their source revision; cleaning does not overwrite them. Use the saved-report menu to reopen recent results, download report JSON, or download individual charts as PNG.
8. **Models:** choose a learning task and model, select a target for supervised tasks, and check features or **Select all except target**. Adjust the split, scaling, seed, and hyperparameters, then **Train model**. Saved model runs can be reopened from the report menu.
9. **Deep learning:** choose an architecture, select numeric features and a target (or upload a CNN image collection), configure training, then **Train neural network**. The live panel shows epoch loss/accuracy and the best completed checkpoint. Saved runs retain their curves, metrics, and outputs.

## Import formats and parsing reference

![Animated import dialog showing automatic parsing and explicit delimiter and encoding choices](docs/media/import-options.gif)

*The import dialog accepts a file or an approved public CSV/Google Sheets URL. This animation demonstrates the parser settings; it does not submit a replacement dataset.*

| Format | Accepted extension | Options / behavior |
| --- | --- | --- |
| Delimited tables | `.csv`, `.tsv`, `.txt` | Detect encoding and delimiter or specify them. Toggle the header row. TSV defaults to tab. TXT can use a bounded regular-expression delimiter. |
| JSON records | `.json` | Array of objects, one object, or a wrapped array selected with `record_key`. Flatten nested objects to depth 0–20; default 3. Remaining nested cell objects/lists become JSON text. |
| Excel | `.xlsx`, `.xls` | First job lists worksheets. Select one or more, then import. Multiple sheets are unioned by column name with `_source_sheet` identifying origins. |
| Parquet | `.parquet` | Reads typed columns after checking metadata against row/column caps. |
| XML | `.xml` | Default XPath `./*`; restricted element paths only. DTD/entity expansion is rejected. |
| SQLite snapshot | `.db` | Lists ordinary tables; select one. Read-only immutable connection; no uploaded database code, extensions, virtual tables or live database connection. |
| SQL dump | `.sql` | Restricted MySQL-style explicit `CREATE TABLE` and literal `INSERT ... VALUES`; choose a parsed table. Statements are parsed as data, never executed. |
| Remote CSV / public Google Sheet | Approved HTTPS URL | Host must be in `IMPORT_URL_HOSTS`. Google Sheets URLs use a numeric `gid` query parameter, default 0. No login, private network, redirect or compressed-response support. |

The current file validator accepts SQLite uploads with **`.db`**; rename/export other SQLite filename extensions before upload. A file selector's extension hint is not the server's format contract. Formats such as ZIP are accepted only through the separate CNN image workflow.

### Parsing examples

| Input shape | Example settings | Explanation |
| --- | --- | --- |
| `name;value` | `{"delimiter":";","encoding":"utf-8","header":true}` | Explicit semicolon-separated text. |
| Headerless values | `{"header":false}` | Treat the first row as observations, not names. |
| Whitespace TXT | `{"delimiter":"\\s+","regex_delimiter":true}` | Split lines on runs of whitespace. Regex TXT is not a CSV quoting parser. |
| `{"records":[{"site":"A","value":2}]}` | `{"record_key":"records","depth":3}` | Select the list inside `records`. |
| Several sheets | `{"sheets":["January","February"]}` | Combine selected worksheets and record source-sheet names. |
| Database/dump | `{"table":"observations"}` | Import the named available table. |
| XML observations | `{"xpath":"./observations/row"}` | Select row elements with a supported element path. |

Column names are normalized to strings and must be nonempty, unique, at most 200 characters, and not `__arg_rowid__`. This reserved internal row ID supports stable previews within a revision and is omitted from normal exports. Dataset row positions reset after a transformation that removes rows. Mixed Python object columns are normalized for Parquet storage.

### Upload lifecycle

1. The authenticated request streams bytes into private storage. An aggregate multipart cap stops oversized uploads before unbounded disk growth.
2. The worker validates extension, reported type, signatures, expansion limits and parser options.
3. Excel/database imports can finish with a **choose worksheet/table** result. The UI submits the selected choices using the owner-scoped saved upload.
4. Successful parsing creates a new Dataset and revision 0, profiles it and points the Workspace to it.
5. A failed import leaves the previously active dataset in place. Original uploads remain private on disk under the retention behavior described below.

## Cleaning operations reference

![Animated walkthrough of missing values, selected numeric columns, and the saved filled revision](docs/media/preparation.gif)

*The sample begins with five missing cells. The selected concentration/temperature columns are filled using the saved mean-fill operation, producing a revision with no missing cells. The walkthrough uses undo/redo of the existing recipe.*

### Missing values

| Operation / method | What it does | Details to check |
| --- | --- | --- |
| Fill → Mean / Median | Fill numeric gaps using each column's mean or median. | Computes from the revision's selected columns, then writes only selected rows. Median is less sensitive to extreme values. Entirely empty numeric columns have no inferable mean/median. |
| Fill → Mode | Fill using the first mode returned for each selected column. | Ties follow the underlying mode ordering; an entirely empty set needs an explicit constant. |
| Fill → Constant | Fill gaps with the supplied value. | Numeric columns require a numeric value; categorical columns can add the new category. |
| Fill → Forward / Backward fill | Carry the preceding/following nonmissing value through gaps. | Uses stored row order, not the current preview sort. Endpoint gaps can remain. |
| Fill → Linear interpolation | Fill numeric gaps between observations. | Interior gaps only; it does not extrapolate beyond both ends. |
| Fill → KNN | Estimate missing numeric values from nearby rows. | 1–50 neighbors; at most 10,000 rows and 50 selected columns; reject entirely empty inputs. Consider feature scales. |
| Fill → Most frequent per group | Fill each column using the mode within a chosen grouping column. | Group column must be outside the selected fill columns. Groups without an observed mode retain gaps. |
| Drop rows with missing → Any | Remove selected rows with at least one gap in selected columns. | Values outside the chosen columns do not decide the removal. |
| Drop rows with missing → All | Remove selected rows only when every selected column is missing. | Check the selected columns before applying. |
| Drop mostly empty columns | Drop selected columns whose missing percentage is strictly greater than the threshold. | Whole columns only; default threshold 50%, permitted 0–100%. A value exactly equal to the threshold is retained. |

### Duplicates and outliers

| Operation | Behavior |
| --- | --- |
| Detect duplicate rows | Flags all occurrences of duplicates on the selected column subset, including the first occurrence. Detection does not create a new data revision. |
| Remove duplicate rows | Keeps the first occurrence and removes subsequent duplicate rows within the selected row scope. |
| Detect outliers → IQR | Flags a row if any selected numeric column is below Q1 − k×IQR or above Q3 + k×IQR. Default k=1.5; permitted 0.1–10. |
| Detect outliers → Z-score | Flags absolute population-standardized deviations above a threshold. Default 3; permitted 0.1–20. Constant columns do not produce a finite deviation score. |
| Detect outliers → Isolation forest | Seeded anomaly detector with median-filled numeric inputs and contamination 0.001–0.5; default 0.05. Needs nonempty numeric input columns and at least two rows; capped at 100,000 rows. |
| Remove flagged rows | Uses the reviewed outlier detection operation for the current revision and writes a new snapshot. An unusual observation is a candidate for review, not automatic evidence of an error. |

### Types, text, encoding and scaling

| Family | Choices | Implementation behavior |
| --- | --- | --- |
| Convert data type | Numeric, String, Datetime, Category | Numeric/datetime conversion fails on incompatible values. Datetimes use optional format or mixed parsing and normalize to UTC. Whole columns only. |
| Clean text | Trim, Lowercase, Uppercase, Title case | Operates on selected string-like values. Convert numeric/datetime columns to string explicitly first. |
| Regex find & replace | Pattern and replacement | Pattern 1–200 characters; replacement ≤1,000; per-substitution timeout 0.05 seconds. No Python evaluation. |
| Strip special characters | Unicode letters/numbers/whitespace retained | Removes punctuation/symbols with bounded regex processing. |
| One-hot encoding | One output per category plus missing indicator | Replaces selected category columns; projected output capped at 500 columns. |
| Label encoding | Ordered integer codes | Categories are sorted by their string representation; missing values remain nullable. Codes do not imply a meaningful numeric distance. |
| Ordinal encoding | Explicit JSON order, e.g. `["low","medium","high"]` | Order must contain every observed nonmissing value exactly once. |
| Target encoding | Category's mean numeric target | Target must be outside the encoded columns. This preparation operation sees the full supplied dataset and can leak target information into later evaluation. |
| Standard scaling | Center by mean; divide by standard deviation | Fits the preparation scaler on the full selected columns. |
| Min–max scaling | Transform to the fitted minimum/maximum range | Extreme values influence the range. |
| Robust scaling | Center/scale using robust statistics | Useful for inspecting data with strong extremes, but does not remove them. |
| Log transform | `log1p(x)` | Every selected value must be greater than −1. |

Preparation transforms change the stored dataset. The ML and deep-learning training pipelines have their own train-only preprocessing; use those fitted preprocessors when measuring holdout performance.

### Column structure

![Drop-column interface with an explicit column selection and Undo reminder](docs/media/09-drop-column.png)

| Operation | Required selection | Result |
| --- | --- | --- |
| Drop selected columns | Explicit column checkboxes; retain at least one column | Removes exactly those columns. Undo restores the preceding snapshot. The screenshot shows configuration before applying. |
| Rename a column | One column | Requires a new unique nonblank name. |
| Split a column | One column and a literal delimiter | Keeps the original and appends `<name>_1`, `<name>_2`, …, up to 20 parts. Rejects output-name collisions. |
| Combine columns | At least two columns, delimiter and new output name | Keeps originals and appends the joined string column; missing pieces become empty strings. |

All structural operations, encodings, conversions and scalings require whole columns: clear row/cell/range selections first. Successful cleaning always creates a snapshot; a rejected operation does not replace the active data.

## History, recipes and exports

![Animated revision history showing the current step moving backward with Undo](docs/media/history.gif)

### How undo/redo works

The Dataset's cursor selects a Revision. Undo and Redo move this cursor to an existing adjacent snapshot; they do not recalculate the operation. A new cleaning step after Undo creates a new branch: later revision records are removed and the new step takes the next revision number. Their private physical files remain until a separate retention process removes them. Reports also retain the immutable revision record ID, so a reused display number does not make an old report current.

### Recipes

**Export recipe** downloads applied operations and the original ordered column schema. **Replay recipe** checks that the incoming dataset matches that schema, then applies each saved operation. A recipe is a list of transformations, not a fitted preprocessing model: medians, target means and scales are recomputed on the new dataset. Cell/row references replay the same positions. Review positional edits before replaying on differently ordered data.

Recipes must contain 1–100 operations. Recipe replay commits the completed sequence atomically at the metadata level: a failed step does not advance the active revision into a partially applied recipe. Intermediate private files can remain for later cleanup. Import options are provenance; the recipe applies cleaning steps to the already-imported matching dataset.

### Choose an export

![Animated export dialog switching between CSV, JSON and Parquet](docs/media/export-formats.gif)

| Export | Use it for | Details |
| --- | --- | --- |
| CSV / TSV | Interoperability with tabular tools | Text representation; separator is comma/tab. Formula-like text is prefixed with an apostrophe for spreadsheet safety. |
| JSON records | Structured interchange preserving text | Dates use ISO representation. No spreadsheet formula prefixing. |
| Parquet | Typed storage and efficient Python/analytics workflows | Preserves table types more faithfully than delimited text; omits the internal preview ID. |
| Excel `.xlsx` | Spreadsheet review | Formula-like text is neutralized. Timezone-aware datetimes export as UTC wall time without timezone metadata. |
| Recipe JSON | Repeat preparation steps | Contains operations and schema, not fitted model weights or the dataset itself. |
| Report JSON / chart PNG | Share an analysis result | JSON includes full report data and Plotly figure descriptions; PNG captures an individual chart. |
| ML `.pkl` / neural `.pt` | Reuse a trained model | Includes model configuration and fitted preprocessing; see the model-specific loading examples. |
| Output CSV | Inspect row-level results | Predictions, assignments, embeddings, neural probabilities, reconstruction errors or latent coordinates, depending on the job. |

Export jobs use the **entire active revision**, not just the currently filtered preview. Clearing a dataset disconnects the active association; it does not purge private stored files or reports.

## Phase 2 analysis guide

![Animated statistical report tour showing correlation and regression diagnostics](docs/media/statistics.gif)

![Correlation report with its selected columns, heatmap and saved-report controls](docs/media/10-correlation.png)

- **Correlation:** Auto chooses a compatible method per pair; override with a named method when appropriate. Pearson measures linear association, Spearman/Kendall measure rank association, point-biserial handles binary/numeric pairs, Phi handles two binary columns, and Cramér’s V handles categorical pairs. Reports document binary coding and incompatible pairs. Pairwise complete observations are used; p-values are unadjusted for multiple comparisons.
- **Regression:** select the numeric target and predictors. Custom models add degree 2/3 polynomials and pairwise interactions through the formula builder. Formulas are generated from internal aliases, so uploaded column names cannot execute code. Forward, backward, and bidirectional selection support AIC, BIC, or nested F-test p-values and preserve term hierarchy. Final inference does not account for selection. All candidates use the same complete rows.
- **Diagnostics:** coefficient tables include standard errors, t-tests, p-values, and confidence intervals. Reports include R², adjusted R², F-tests, AIC/BIC, Shapiro–Wilk, Anderson–Darling critical values, D’Agostino–Pearson, Breusch–Pagan, White, Durbin–Watson, Ljung–Box, and configurable VIF flags. The KS normality check uses [Lilliefors calibration](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.lilliefors.html) because distribution parameters are estimated. Autocorrelation follows dataset row order, not display sorting. Undefined or inapplicable statistics show a dash with explanatory notes.
- **ANOVA:** ordinary one/two-way models assume independent groups and equal variances. Two-way includes interaction and uses type II sums of squares. Welch permits unequal variances. Repeated measures accepts one within-subject factor in long format, requires one observation per subject/condition, and excludes incomplete subjects. Reports show [sphericity and Greenhouse–Geisser correction](https://pingouin-stats.org/generated/pingouin.rm_anova.html). Tukey is available for ordinary ANOVA only; two-way Tukey pools the other factor and is not a simple-effects test.
- **Plots:** histogram, box, violin, scatter, pair, Pearson heatmap, categorical counts, and line charts. Line charts sort by X. Regression includes residuals vs fitted, normal Q–Q, scale–location, Cook’s distance, and residuals vs leverage. Large point/distribution plots are sampled and explicitly labeled; statistical tables use all applicable observations.

### Correlation methods explained

| Method | Compatible input | What the coefficient describes |
| --- | --- | --- |
| Pearson | Two numeric columns | Strength and direction of a linear relationship; sensitive to influential observations. |
| Spearman | Two numeric columns | Rank-based monotonic association, including nonlinear monotonic relationships. |
| Kendall | Two numeric columns | Agreement between pairwise rank orderings. |
| Point-biserial | One binary and one numeric column | Numeric differences associated with a two-level category; the report states its 0/1 coding. |
| Phi | Two binary columns | Signed association between the report's coded binary variables. |
| Cramér's V | Two categorical columns | Nonnegative contingency-table association; this implementation uses the uncorrected coefficient. |
| Auto | Selected column pairs | Checks binary/binary first, then binary/numeric, numeric/numeric and categorical/categorical. Unsupported mixed pairs receive an explanation. |

Reports include coefficients, p-values, paired observation counts and method notes. Constant columns or insufficient paired rows produce unavailable entries with explanations. Different pairs may use different complete rows. Correlation does not establish causation; small expected contingency counts can make asymptotic p-values unreliable.

### Descriptive summaries and EDA

Descriptive reports summarize selected columns and missingness before formal modeling. EDA provides histogram, box, violin, scatter, pair plot, Pearson heatmap, categorical count and line charts. Choose plots according to the question: distributions for spread/skew, scatter for a pair's relationship, counts for category frequencies and ordered line plots for a progression. Plot sampling is a display limit; notes identify it, and statistics use their documented analysis rows.

### Regression outputs and assumption checks

| Output / diagnostic | Interpretation |
| --- | --- |
| Coefficient, standard error, t-statistic, p-value, confidence interval | Estimated change associated with a term, conditional on the fitted design; intervals/inference assume an appropriate model. |
| R² / adjusted R² | In-sample explained variation; adjusted R² accounts for model size. These are not out-of-sample prediction scores. |
| F-statistic / p-value | Joint model significance for the fitted OLS design. |
| AIC / BIC | Penalized fit criteria for comparing candidate models on the same response and observations. Lower values are preferred within that comparison. |
| Shapiro–Wilk / D'Agostino–Pearson | Residual normality checks; minimum sample requirements differ. |
| KS with Lilliefors / Anderson–Darling | Normality assessment accounting for fitted parameters or reporting reference critical values. Anderson–Darling does not return a p-value here. |
| Breusch–Pagan / White | Evidence that residual variance depends on predictors; White expands the auxiliary design with squares/cross-products. |
| Durbin–Watson / Ljung–Box | Residual serial dependence in dataset order; meaningful ordering matters. |
| VIF | Predictor redundancy within the regression design; large values signal unstable separate coefficient interpretation. |
| Cook's distance / leverage | Observations that can exert disproportionate influence or occupy unusual predictor positions. |

Custom regression permits selected interactions and degree-2/3 polynomial terms. The implementation constructs formulas from internal safe names. Forward, backward or bidirectional stepwise selection uses AIC, BIC or nested F-tests; reported inference does not adjust for the selection process. Near-perfect fits can make residual diagnostics meaningless, and reports explain unavailable values instead of turning them into zero.

### ANOVA choices

| Choice | Data layout | Main distinction |
| --- | --- | --- |
| One-way | Numeric outcome and one independent group factor | Tests mean differences across groups with the ordinary equal-variance model. |
| Two-way | Numeric outcome and two factors | Includes both main effects and their interaction; uses type II sums of squares. |
| Welch | Numeric outcome and one independent group factor | Allows unequal group variances. |
| Repeated measures | Subject ID, one within-subject factor and outcome in long format | Requires one observation per subject/condition and retains complete subjects. Includes sphericity/correction information. |
| Tukey HSD | Ordinary independent-groups ANOVA | Pairwise group comparison after the ordinary model; two-way Tukey pools the other factor rather than testing simple effects. |

Read effect estimates and assumptions alongside p-values. Repeating many tests creates additional false-positive opportunities; the correlation p-values and exploratory analyses are not a universal multiple-testing correction system.

## Phase 3 modeling guide

![Animated modeling walkthrough from model controls to a saved regression report and binary threshold exploration](docs/media/modeling.gif)

![Classical model report with train/test results and reusable artifacts](docs/media/13-model-report.png)

| Task | Models |
| --- | --- |
| Regression | Linear, Ridge, Lasso, ElasticNet, polynomial (degree 2/3), SVR, decision tree, random forest, gradient boosting, XGBoost, LightGBM, KNN |
| Classification | Logistic regression, decision tree, random forest, SVM, KNN, Gaussian/Multinomial naive Bayes, gradient boosting, XGBoost, LightGBM |
| Clustering | K-Means with elbow/silhouette helper, hierarchical/agglomerative with dendrogram, DBSCAN |
| Reduction | PCA with scree/variance report, t-SNE, UMAP |

- **Preprocessing and splits:** select 1–50 features. Numeric missing/non-finite values are median-imputed; categories receive a missing token and one-hot encoding, capped at 32 outputs per input feature. Unknown categories become all zeros. Scaling is configurable. Supervised models split first and fit every preprocessing step only on training rows. Search refits preprocessing inside each CV fold. Missing targets are excluded, and classification supports 2–30 classes with optional stratification. Datetime features require explicit conversion before modeling. Earlier cleaning operations have already seen the entire dataset, so avoid target encoding or other learned cleaning on the full dataset before evaluating a model. [scikit-learn guidance on leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- **Search:** supervised grid/random search uses user-specified candidate arrays, 2–5 training-only CV folds, and at most 20 candidates. Classification maximizes accuracy; regression maximizes negative RMSE. Failed candidates are listed. The winner is fitted on the training split, then evaluated once on the held-out set. Reset defaults restores the selected model's parameter defaults and clears candidate ranges. Unsupervised models have direct parameter controls; K-Means provides a separate cluster-count helper.
- **Evaluation:** regression shows RMSE, MAE, R², and actual/predicted plots. Classification shows accuracy, precision, recall, F1, ROC-AUC, and confusion matrix; multiclass metrics are support-weighted. Feature importance uses original-column permutation on up to 500 held-out rows with three repeats, so it also works for SVMs, KNN, and polynomial models. Importance can be negative; correlated features can obscure it.
- **Binary cutoff:** the live slider uses precomputed counts from every held-out row at thresholds 0.00–1.00, updating confusion counts, accuracy, precision, recall, and F1 immediately. Reports name the positive class. This exploration does not mutate the fitted model; choose a prediction threshold separately when scoring an upload. If you select a threshold using the test results, validate it on fresh data before treating the result as an unbiased estimate.
- **Clustering/reduction:** these fit all selected rows. K-Means evaluates k=1–10 where feasible without overriding the chosen k; silhouette samples up to 1,000 observations. DBSCAN noise is excluded from silhouette. Hierarchical dendrograms show the last 30 merged groups. Clustering plots use a separate PCA display projection. Full row assignments/embeddings are available as CSV. t-SNE, DBSCAN, and hierarchical clustering have no supported out-of-sample prediction here; their fitted artifacts and outputs remain downloadable. PCA and [UMAP support transforming new data](https://umap-learn.readthedocs.io/en/latest/transform.html).
- **Artifacts and prediction:** download the `.pkl` bundle containing the fitted pipeline, feature schema, class mapping, source revision, and package versions. The pipeline stays fitted on training data; it is not refitted on test rows. New-file scoring is a background job, validates required feature names/types, ignores extra columns, and preserves the active workspace dataset. Supported parsing formats match ingestion; Excel/SQL files require sheet/table parsing options. Download the complete scored output as CSV.

Model artifacts are created by this server and scoped to their owner. The server never accepts uploaded pickle/joblib files. Only load trusted downloaded model bundles, using the recorded package versions: [pickle-based persistence can execute code when loaded](https://scikit-learn.org/stable/model_persistence.html).

### Understanding the model families

| Family | What it learns / useful distinction |
| --- | --- |
| Linear / polynomial regression | A linear combination of features; polynomial regression expands the preprocessed feature terms before fitting. |
| Ridge / Lasso / ElasticNet | Regularized linear models. Ridge shrinks coefficients, Lasso can shrink some to zero, ElasticNet combines the penalties. |
| SVR / SVM | Margin-based regression/classification with configurable kernels; row caps reflect their computational cost. |
| Decision tree | A sequence of feature-based splits; depth and minimum leaf sizes constrain complexity. |
| Random forest | An ensemble of randomized trees; tree count controls ensemble size. |
| Gradient boosting / XGBoost / LightGBM | Additive ensembles that improve on previous errors; learning rate and tree count interact. |
| K-nearest neighbors | Predictions from nearby fitted observations; scaling and neighborhood size matter. |
| Logistic regression | A linear classification model with probability outputs and regularization. |
| Gaussian / Multinomial naive Bayes | Class-conditional feature models with different distribution assumptions. Multinomial NB forces train-fitted min–max scaling and clipping here to keep values nonnegative. |
| K-Means | Assigns rows to a chosen number of centers; the elbow/silhouette helper supports exploration. |
| Hierarchical clustering | Builds clusters by successive merges; the report provides a truncated dendrogram. |
| DBSCAN | Density-based clusters and noise points; does not require a cluster count and can leave rows unassigned as noise. |
| PCA | Linear projection ranked by explained variance; supports transforming new observations. |
| t-SNE / UMAP | Nonlinear embeddings for exploration. Distances in a display need careful interpretation; UMAP supports new-row transforms here, t-SNE does not. |

The exact **28 model keys, defaults, allowed values and ranges** are listed in the [complete generated model parameter reference](docs/model-parameters.md). This reference is generated directly from the same catalog that renders the model controls.

### Training and evaluation sequence

```mermaid
flowchart LR
    A[Selected features and target] --> B[Remove missing targets]
    B --> C[Seeded train/test split]
    C --> D[Fit preprocessing on training rows]
    D --> E[Fit estimator or training-only CV search]
    E --> F[Evaluate held-out test rows]
    F --> G[Save report and fitted pipeline]
    G --> H[Reuse pipeline on uploaded new rows]
```

The supervised test fraction defaults to 0.20 and can be 0.10–0.50. Classification can use stratification so splits preserve class proportions where sample counts permit it. Numeric preprocessing is median imputation plus optional standard/min–max/robust scaling; categorical features are encoded by the fitted pipeline. Unsupervised tasks fit the selected full dataset and therefore do not report a supervised holdout score.

Search uses JSON candidate arrays, for example `alpha: [0.1, 1, 10]`. Grid search evaluates every allowed combination; random search samples a bounded number with the selected seed. The report lists candidate scores and failures. Neither approach searches arbitrary Python parameters outside the whitelisted controls.

### Metrics and classification cutoff

| Metric | Meaning |
| --- | --- |
| RMSE | Square root of mean squared prediction error; emphasizes larger errors, in target units. |
| MAE | Mean absolute prediction error, in target units. |
| R² | Improvement over predicting the test outcome mean; it can be negative. |
| Accuracy | Fraction of correctly classified test rows. |
| Precision | Of rows predicted positive, the fraction actually positive; multiclass reports use documented averaging. |
| Recall | Of actual positive rows, the fraction recovered. |
| F1 | Harmonic mean of precision and recall. |
| ROC-AUC | Ranking discrimination across thresholds; unavailable when class/support conditions prevent calculation. |
| Confusion matrix | Actual versus predicted class counts. |
| Silhouette | Cluster separation/cohesion diagnostic; requires a meaningful set of nontrivial clusters. |
| Explained variance | Variation retained by a PCA projection; not predictive accuracy. |

![Animated binary classification cutoff showing how metrics and confusion counts change](docs/media/classification-cutoff.gif)

*The saved model stays fixed while the report's probability cutoff changes. This is test-set exploration; it does not retrain the model or establish a new unbiased score.*

To score a new file, reopen a compatible saved model, choose **Prediction data file**, optionally provide parsing JSON and a binary probability cutoff, then **Score uploaded data**. Required feature names must exist; extra columns are ignored. The active preparation dataset is not replaced. This workflow is unavailable for t-SNE, DBSCAN and hierarchical clustering because their fitted artifacts do not expose a supported out-of-sample operation here.

To score a trusted exported bundle within this project environment:

```python
import joblib
import pandas as pd
from workspace.ml.prediction import predict

bundle = joblib.load("trained-model.pkl")  # Only a model you trust.
result = predict(bundle, pd.read_csv("new-data.csv"), threshold=0.5)
result.to_csv("predictions.csv", index=False)
```

Recipes describe transformations, not fitted ML preprocessors: statistics, encodings, and scaler parameters are recomputed on the new dataset. Explicit cell/row selections replay the same positions. Target encoding uses the entire supplied dataset's target means; fit it only on training data to avoid leakage in later modeling.

## Phase 4 deep-learning guide

![Animated neural workflow showing configuration, live epoch curves, and saved training results](docs/media/deep-learning.gif)

![Deep-learning report with architecture controls, held-out metrics, and checkpoint output](docs/media/18-deep-report.png)

| Architecture | Input and output |
| --- | --- |
| MLP / ANN | Numeric tabular features; regression or 2–20-class classification |
| CNN | A separate ZIP or folder containing `class_name/image.png` (optional common parent folder); image classification |
| RNN / LSTM / GRU | One continuous sequence of numeric rows; a preceding window predicts the next row's numeric target or class |
| Autoencoder | At least two numeric features; reconstruction error, anomaly flags, and a lower-dimensional latent representation |

- **Configuration:** dense layer widths, ReLU/Tanh/GELU, Adam/SGD/RMSprop, learning rate, batch size, epochs, dropout, seed, split fractions, and optional early stopping with patience. Sequence networks add window length, recurrent layers/units, and optional order column. CNNs add convolution channels; autoencoders add latent dimensions and anomaly-error quantile. Recurrent cells retain PyTorch's supported internal activations; the selected activation also controls the dense head.
- **Splits and preprocessing:** validation and test each reserve 10–30% of rows; the remainder trains the network. MLP/CNN/autoencoder splits are seeded; classification is stratified. Sequence splits are chronological, with windows overlapping only within each partition. An explicit order column must contain unique, nonmissing values; otherwise current dataset order is used. Do not combine independent subjects/series in one run. Missing sequence targets are rejected. Tabular features use medians/scales fitted only on training rows. Regression targets are standardized from training rows; final metrics return to original units. Encode categorical inputs before training, and avoid any full-dataset learned preprocessing that could leak holdout information.
- **Validation and outputs:** validation loss selects the saved weights and triggers early stopping; the held-out test partition is evaluated afterward. Regression reports RMSE, MAE, R² and predictions; classification reports accuracy, precision, recall, F1, ROC-AUC and confusion matrix. Autoencoder error is measured in standardized feature space; anomaly flags use a quantile of training reconstruction error and are heuristic. All test outputs download as CSV, including original row numbers (CNN outputs also include the original image paths), probabilities or latent coordinates. Saved report JSON includes configuration, history and source metadata. Plotly figures export as PNG.
- **Images:** validation and decoding run in the deep worker without extracting archives to disk. PNG/JPEG/WebP/BMP headers must match extensions. Images become center-cropped RGB 64×64 tensors. Exact duplicates after resizing are removed before splitting; conflicting labels are rejected. Use 12–2,000 images, 2–20 classes, and at least six distinct images per class. ZIPs reject traversal, symlinks, encryption, unexpected files, duplicate paths and excessive expansion. Folder uploads use the same checks and aggregate upload cap. Image collections remain separate from the active tabular dataset.
- **Live progress and checkpoints:** epoch history and best validation weights persist after each completed epoch. Reload the page and reopen Deep learning to resume watching. A failed/interrupted run retains its last completed checkpoint. `.pt` checkpoints contain tensors, network configuration, training-only preprocessing, class mapping and source metadata; optimizer state is not saved, so these are inference checkpoints rather than resumable training sessions. The browser currently scores new uploaded files with classical models; neural checkpoints can be used from Python.
- **CPU/CUDA:** Auto chooses CUDA if the worker can access it, otherwise CPU. CUDA required fails clearly when unavailable. Local validation used PyTorch 2.8.0+cpu; CUDA execution and Docker GPU passthrough have not been tested on this machine. To enable CUDA, install the compatible **2.8.0** wheel from [PyTorch's official installation commands](https://pytorch.org/get-started/previous-versions/#v280), verify `python -c "import torch; print(torch.cuda.is_available())"`, then restart the deep worker or `start.bat`. Hosted deployments can place the `deep` queue worker on a GPU host sharing the same database, Redis and private media. The default Compose setup does not request GPU access; with [NVIDIA Container Toolkit and GPU reservations configured](https://docs.docker.com/compose/how-tos/gpu-support/), use `docker compose -f compose.yaml -f compose.gpu.yaml up --build`.

Example: use an exported MLP regression checkpoint in this project environment (the helper uses [PyTorch's `weights_only=True` loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)):

```python
import numpy as np
import pandas as pd
import torch
from workspace.deep.networks import restore_checkpoint
from workspace.deep.data import transform

model, checkpoint = restore_checkpoint("best-checkpoint.pt")
pre = checkpoint["preprocessing"]
new = pd.read_csv("new-data.csv")
values = new[pre["features"]].to_numpy(dtype=float)
x = torch.from_numpy(transform(values, pre["scaler"]))
with torch.no_grad():
    standardized = model(x).numpy()[:, 0]
predictions = standardized * pre["target_scaler"]["scale"] + pre["target_scaler"]["mean"]
```

For classification, apply softmax to logits and map indices with `pre["classes"]`. Sequence inputs have shape `[batch, window_length, features]`; CNN inputs have shape `[batch, 3, 64, 64]` scaled to `[0,1]`, using the same crop/resize process. Autoencoders reconstruct standardized features; `model.encoder(x)` returns latent coordinates. The server does not accept uploaded checkpoints.

### Every neural control

| Control / API key | Default | Allowed values / effect |
| --- | --- | --- |
| Architecture / `architecture` | `mlp` | `mlp`, `cnn`, `rnn`, `lstm`, `gru`, `autoencoder`. CNN forces classification; autoencoder forces reconstruction. |
| Task / `task` | `regression` | Regression or classification for MLP/sequence networks. |
| Features / `features` | No UI selection | 1–50 existing distinct numeric columns; target must be separate for supervised tasks. |
| Target / `target` | No UI selection | Existing numeric regression target or 2–20-class classification target. |
| Dense widths / `hidden_layers` | `[64,32]` | 1–4 widths, each 4–512. UI accepts comma-separated values. |
| Activation / `activation` | `relu` | ReLU, Tanh or GELU for dense layers and CNN blocks. RNN cells support ReLU/Tanh; LSTM/GRU use built-in gates. |
| Optimizer / `optimizer` | `adam` | Adam, SGD or RMSprop. |
| Learning rate / `learning_rate` | `0.001` | 0.00001–0.1. Nonfinite/divergent losses fail with a useful message. |
| Batch size / `batch_size` | `32` | 4–256; actual final batches can be smaller. |
| Epochs / `epochs` | `30` | 1–200, also subject to sample-visit and time caps. |
| Dropout / `dropout` | `0.1` | 0–0.7. Applied during training; disabled for evaluation. Recurrent inter-layer dropout needs more than one recurrent layer. |
| Early stopping / `early_stopping` | `true` | Stop after patience epochs without sufficient validation-loss improvement. Best validation weights are retained even when stopping is disabled. |
| Patience / `patience` | `5` | 1–30 epochs. Improvement must exceed 0.000001. |
| Validation / `validation_size` | `0.2` | 0.1–0.3 of original rows/images before sequence window construction. |
| Test / `test_size` | `0.2` | 0.1–0.3; not used for early stopping. |
| Seed / `seed` | `42` | Integer 0–2,147,483,647; seeds splits and training RNGs. Different devices/software may still produce differences. |
| Device / `device` | `auto` | Auto, CPU, CUDA required. Auto falls back to CPU. |
| Sequence order / `order_column` | Current dataset order | Optional existing unique nonmissing column; sorting happens in the worker. |
| Window / `sequence_length` | `10` | 2–100 preceding rows per prediction. Each split requires at least window length + 2 raw rows. |
| Recurrent depth / `recurrent_layers` | `1` | 1–3 layers. |
| Recurrent width / `recurrent_units` | `32` | 4–256 units. |
| CNN blocks / `conv_channels` | `[16,32]` | 1–4 blocks, 4–128 output channels each. Each block uses 3×3 convolution, activation, pooling and dropout; global pooling feeds the dense head. |
| Collection / `image_id` | None | UUID of a successful image-validation job belonging to the current workspace. |
| Latent dimensions / `latent_dim` | UI: `1`; API omitted: `2` | 1–32, strictly fewer than input features. Controls the autoencoder bottleneck. |
| Error quantile / `anomaly_quantile` | `0.95` | 0.8–0.999 of training reconstruction errors; test errors above it are flagged. |

### Sequence example

For one ordered series of 120 rows, validation/test fractions of 0.2 create 72 training, 24 validation and 24 test rows. A window length of 4 then gives 68, 20 and 20 prediction windows respectively. The first test window uses only the first four rows inside the test partition and predicts its fifth row. No training or validation context crosses into it. Preview sorting alone does not set sequence order.

### CNN collection structure

```text
images.zip
└── optional-parent/
    ├── class_a/
    │   ├── image_01.png
    │   └── ... at least six distinct images
    └── class_b/
        ├── image_01.jpg
        └── ... at least six distinct images
```

Use the class directory names as labels. Do not include README files, nested category trees or unrelated files in the archive. Exact duplicates are removed after the 64×64 crop/resize, so the remaining class counts must still satisfy split requirements. Folder upload depends on the browser supporting directory selection; ZIP is the portable alternative.

### What happens during an epoch

Training batches are shuffled with a seeded generator, loss is calculated, gradients are backpropagated and clipped to norm 1, and the optimizer updates the weights. Classification uses cross-entropy; regression and reconstruction use mean squared error. Validation runs in evaluation mode without gradients. A lower validation loss saves a CPU state-dictionary checkpoint atomically, then the job record receives the new epoch history and progress message. Final test evaluation loads the best weights, not necessarily the last epoch's weights.

Training loss is averaged from minibatches while weights are changing and dropout is active; validation loss is measured after the epoch with dropout disabled. Their difference is useful to inspect, but they are not calculated under identical conditions. Neural training is from scratch; pretrained networks, transfer learning, automatic neural architecture search and distributed training are not included.

## Limits and behavior

| Setting / feature | Behavior |
| --- | --- |
| `MAX_UPLOAD_MB` | 200 MB by default. All accepted files run asynchronously; oversized multipart bodies stop in the upload handler. Multi-GB files are outside scope. |
| `MAX_ROWS`, `MAX_COLUMNS` | 1,000,000 rows / 500 columns by default. XLSX expanded data is capped at 4× upload size. |
| SQL dumps | At most 20 MB. MySQL-style `CREATE TABLE` and literal `INSERT ... VALUES` only. No queries, expressions, `SET`, procedures, `COPY`, or arbitrary execution. Export unsupported dumps as CSV. |
| SQLite | Immutable, read-only access with extension loading disabled. Ordinary tables only; no views, virtual tables, or live external database connectors. |
| URL imports | Exact `IMPORT_URL_HOSTS` allowlist; HTTPS port 443, public DNS addresses, IP-pinned TLS, CSV content types, no proxies/redirects/compressed responses. Google Sheets must be public. Upload restricted or redirecting sources as files. |
| XML / regex | Entity expansion disabled; XPath restricted to element paths. Regex length caps, per-match timeouts, and worker time limits. Regex TXT is line-based and does not implement CSV quoting. |
| Preview | Maximum 100 rows. Text over 2,000 characters is marked as truncated; inline editing is disabled for that cell, and exports retain the full value. |
| Missing values | Mode cannot infer entirely empty columns; interpolation fills interior gaps. Use constant fill when no observations exist. |
| Expensive cleaners | KNN: at most 10,000 rows / 50 columns. Isolation forest: at most 100,000 rows. One-hot output: at most 500 columns. |
| Analyses | At most 30 applicable columns per report; all analysis jobs run asynchronously. Cramér’s V: at most 50 levels per column. |
| Regression | At most 20 predictors / 60 terms / 20 million design cells. Stepwise: at most 12 terms and 200 candidate fits. White auxiliary design: at most 10 million cells and more observations than auxiliary columns. |
| ANOVA | 2–30 levels per factor; independent groups need at least two observations each. Two-way needs two observations per cell, at most 100 cells and 10 million design cells. Repeated measures needs at least three complete subjects. |
| Plot size | At most 5,000 plotted rows; regression plots retain the 200 largest Cook’s distances. Pair plots: at most six numeric columns; distribution plots: at most 12. Shapiro and Anderson use a deterministic sample of at most 5,000 residuals. |
| Reports | Latest 30 successful statistical reports, 30 ML/prediction jobs and 30 deep runs appear in their menus. Older reports remain available by their private job URLs; JSON includes parameters, tables, figures, warnings, and source revision. |
| ML training | 8–50,000 rows overall; supervised models require at least 12 labeled rows and two held-out rows. SVR/SVM and UMAP: at most 10,000 rows; DBSCAN: 5,000; t-SNE: 3,000; hierarchical: 2,500. Encoded/polynomial designs: at most 2,000 features and 10 million cells. |
| ML search | Supervised tasks only; at most 20,000 training rows, 20 candidates, 2–5 CV folds, and 1–10 listed values per searched parameter. Every estimator uses bounded key hyperparameters and a single estimator thread. |
| ML plots / scoring | Predictions: at most 50,000 rows and 10 million encoded cells. Held-out plots: 2,000 points; embeddings: 3,000 points; cluster legends group beyond the 30 largest. Full results remain in CSV. |
| Rate limits | 20 upload requests/hour and 120 jobs/hour per user using Redis; one active job per workspace. DRF throttles are approximate under concurrency; add ingress limits when deploying. |
| Time limits | Soft 300 / hard 330 seconds on supported Linux worker pools. Files must fit worker memory after expansion. |
| Neural training | 20–20,000 tabular rows; 1–50 numeric features; 1–200 epochs; at most 2 million training sample visits, 10 million sequence input cells, and 2 million parameters. A 15-minute training-loop deadline preserves the best completed checkpoint. The deep task declares 30/31-minute soft/hard limits, which require a pool supporting them; the default solo deep worker needs deployment-level process/container limits. |
| Image decoding | Aggregate upload limit applies; ZIP expansion ≤400 MB, each image ≤12 MB, ≤16 megapixels and ≤4,096 pixels per side; compression ratio ≤200. All images must fit worker memory. |
| Spreadsheet exports | CSV/TSV/XLSX prefix formula-like text with an apostrophe. JSON/Parquet preserve text values. Excel datetime cells use UTC with timezone information removed because Excel cannot store it. |
| Retention | Clear/replacement removes the active association, not the physical files. Old snapshots, sources, exports, and failed recipe intermediates stay private on disk. Automated retention/quotas are not implemented. |

Content types are cross-checked against extensions and file signatures. SQL text never enters a database execution API. User preview filters are bound parameters, and identifiers are checked against the dataset schema. Uploaded data has no public media URL.

## Architecture

```mermaid
flowchart TB
    UI[Django templates / Alpine / HTMX / Plotly] -->|Session + CSRF| API[Django REST API]
    UI -->|Bounded page / filter / sort| PREVIEW[DuckDB preview]
    API --> DB[(Metadata database)]
    API --> BROKER[Redis broker / local memory broker]
    BROKER --> NORMAL[celery queue: preparation / statistics / ML]
    BROKER --> DEEP[deep queue: images / PyTorch]
    NORMAL --> FILES[(Private media storage)]
    DEEP --> FILES
    DEEP --> GPU[CUDA if available / CPU fallback]
    PREVIEW --> FILES
    NORMAL --> DB
    DEEP --> DB
    UI -->|Poll job state / download owner-scoped artifacts| API
```

### Request and worker responsibilities

The web process handles authentication, input validation, upload streaming, job submission, bounded previews and authorized downloads. Workers perform expensive parsing, cleaning, profiling, statistics, fitting, neural training and exports. A preview is a bounded read-only DuckDB query over the current Parquet file; it is not a dataframe serialized into a cookie or Django session.

`run_job` atomically claims a queued Job before executing it. Re-delivery of an already-claimed job therefore does not apply the same operation twice. Database uniqueness allows only one queued/running job per workspace. This prevents a second user action in that workspace from changing a dataset under an active task; other workspaces can still submit independent jobs.

The `deep` task routes to a dedicated queue, making it possible to place neural work on a GPU-capable worker while ordinary preparation stays on CPU workers. The local launcher uses one embedded worker for both queues, so it processes jobs serially. The Compose ordinary worker uses two processes; its deep worker uses a single solo process.

### Metadata model

```mermaid
erDiagram
    USER ||--|| WORKSPACE : owns
    WORKSPACE ||--o{ DATASET : contains
    WORKSPACE ||--o{ UPLOAD : receives
    WORKSPACE ||--o{ JOB : runs
    DATASET ||--|{ REVISION : snapshots
```

| Model | Important fields | Why it exists |
| --- | --- | --- |
| Workspace | `owner`, nullable `active_dataset` | Separates user ownership from the currently selected dataset. Owner is one-to-one with a Django account. |
| Upload | UUID, workspace, original name, private path, MIME, timestamp | Preserves a source for parsing/worksheet selection and image validation. |
| Dataset | UUID, workspace, name, cursor, original schema, timestamp | Tracks one imported table and the revision selected by undo/redo. |
| Revision | Dataset, revision number, private Parquet path, operation JSON, profile JSON, timestamp | Stores a snapshot plus its explanation and cached profile. Dataset/number is unique. |
| Job | UUID, workspace, kind, status, progress, message, payload JSON, result JSON, artifact path, timestamps | Durable submission, progress and artifact lookup. Conditional uniqueness enforces one active job per workspace. |

Metadata does not contain the full tabular dataset. File names inside each workspace's private directory use generated UUIDs. The backend validates ownership through the related workspace rather than treating possession of a UUID as authorization.

### Job states

```mermaid
stateDiagram-v2
    [*] --> queued: Valid request and workspace slot
    queued --> running: Worker atomically claims job
    queued --> failed: Broker submission failure
    running --> succeeded: Artifacts and metadata saved
    running --> failed: Validation / computation / timeout failure
    succeeded --> [*]
    failed --> [*]
```

An API `202` response means the job was accepted, not that its analysis passed or training finished. Poll the returned Job URL until a terminal status appears. Progress percentages describe steps, not a reliable time estimate. A successful initial workbook/table-discovery job can request another user choice before a dataset exists. For neural jobs, the result evolves with each epoch and can contain a checkpoint URL before completion.

### Private files and artifact relationships

| File | Producer | Contents / association |
| --- | --- | --- |
| Uploaded source | Ingest / image upload request | Original source bytes, private to the workspace. |
| Revision `.parquet` | Import / cleaning / recipe worker | Full normalized table plus internal preview row ID. |
| Detection `.parquet` | Duplicate/outlier detection | Flagged row IDs used by preview for the matching dataset/revision. |
| Report `.json` | Analysis / ML / deep / prediction | Shared report structure: title, metrics, tables, Plotly figures, notes, warnings, parameters and source. |
| Model `.pkl` | Classical training | Fitted preprocessing/estimator, feature schema, source and package versions. Uses the report file's stem. |
| Checkpoint `.pt` | Neural training | Tensor state dictionary and primitive metadata; best validation checkpoint is saved during training. |
| Image `.npz` + `.json` | Image validation | Deduplicated image tensors/labels plus collection manifest. No arbitrary pickle loading. |
| Output `.csv` | Model/deep/prediction | Complete row-level outputs where supported. |
| Export file | Export job | Selected output format for the current complete dataset revision. |

### Security and correctness boundaries

- Django session authentication and CSRF protection apply to browser/API mutations. There is no API-token authentication interface in this release.
- Data/report/model/checkpoint lookups are scoped to the signed-in owner's workspace. `MEDIA_ROOT` has no public media-serving URL.
- Inputs use approved operations, parser options, estimator constructors and bounded parameters. There is no arbitrary code/eval, shell, SQL execution or uploaded model-deserialization endpoint.
- Remote imports validate allowed hosts and public DNS addresses, pin the validated connection address, and verify TLS for the requested host. No private-sheet connector credentials are stored.
- CSV/TSV/XLSX exports neutralize formula-like strings; JSON and Parquet preserve original text.
- Revision identity is rechecked when the worker starts. Reports retain source identity and do not mutate preparation snapshots.
- Numerical reports convert nonfinite/undefined statistics to JSON-safe unavailable values with relevant notes where implemented.
- Limits reduce accidental excessive workloads, but they are not a deployment-wide RAM/disk quota. Retention, backups, ingress limits and resource enforcement remain operational responsibilities.

## Configuration reference

The base settings read [`.env.example`](.env.example)'s variables through `django-environ`. `scripts/init_env.py` creates a random secret in `.env` without overwriting an existing file or printing the secret. Never put `.env`, private datasets, session cookies or trained artifacts containing private data in Git.

| Variable | Default / example | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | Required; generated by setup | Django signing secret; keep stable for a running installation and private. |
| `DEBUG` | `True` in development example | Enables development behavior. Use `False` with a properly configured deployment. |
| `DATABASE_URL` | `postgres://arg:arg_local@127.0.0.1:5432/arg` in example | Metadata database connection for base settings; launcher overrides it with its dedicated SQLite configuration. |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Broker, result-backend configuration and Django cache in service mode. |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Allowed HTTP Host headers. Local mode pins loopback names. |
| `MEDIA_ROOT` | `media` | Private files directory. Local mode uses `media/local/datasets/`. |
| `MAX_UPLOAD_MB` | `200` | Aggregate multipart file-byte and accepted source-size limit. |
| `MAX_ROWS` | `1000000` | General ingestion row cap; model-specific limits can be lower. |
| `MAX_COLUMNS` | `500` | General ingestion column cap; operation/model-specific limits can be lower. |
| `IMPORT_URL_HOSTS` | `docs.google.com` | Comma-separated exact permitted remote-import hostnames. |

Additional fixed settings include 2 MB non-file request data, 1 MB upload-memory threshold, at most 2,000 uploaded files per request, 20 uploads/hour and 120 jobs/hour per user, and disabled Celery publish retries. Job results are stored in the application's Job records; the UI does not read a Celery result backend directly.

Compose supplies service-internal database/Redis hostnames and shares media among web/workers. Its development database password is an example local credential. Changing `.env` does not override every value explicitly set by Compose; inspect [compose.yaml](compose.yaml) before deployment. Likewise, `project/local_settings.py` deliberately overrides database, broker, cache, cookie names and loopback settings for the launcher.


## API

All `/api/` routes require an authenticated Django session. JSON mutations require a CSRF header, while uploads use multipart bodies. Start from `/api/state/` to discover the current dataset/revision and owner-visible recent artifacts. State returns the latest 10 jobs, 30 successful statistical reports, 30 successful ML/prediction jobs, 30 successful deep runs and 20 successful image collections.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/state/` | Active dataset, cached profile, history, recent jobs and analyses |
| `POST /api/ingest/` | Multipart file/options, owner-scoped upload selection, or URL |
| `GET /api/preview/` | `page`, `page_size`, `q`, `filters` (JSON contains-map), `sort`, `direction`, optional `detection` ID |
| `POST /api/jobs/` | `kind`, current `dataset_id`, current `revision`, and operation/recipe/export parameters or `analysis` object (`type`, `columns`, analysis-specific options) |
| `GET /api/jobs/<id>/` | Queued/running/succeeded/failed status, progress, result |
| `GET /api/jobs/<id>/download/` | Owner-only completed export or report JSON |
| `GET /api/jobs/<id>/report/` | Owner-only completed analysis report for rendering |
| `GET /api/models/catalog/` | Model names, tasks, key hyperparameters, defaults, choices, and bounds |
| `POST /api/jobs/` with `kind: "model"` | Current dataset/revision and `model: {model, features, target, params, test_size, stratify, seed, scaling, search}` |
| `GET /api/models/<id>/download/` | Owner-only trained `.pkl` bundle |
| `GET /api/models/<id>/output/` | Complete cluster assignments, embedding, or prediction CSV |
| `POST /api/models/<id>/predict/` | Multipart `file`, optional parsing `options` JSON and binary `threshold`; returns a queued prediction job without replacing the dataset |
| `POST /api/deep/images/` | Multipart `archive` ZIP, or `images` files plus `paths` JSON array; queues image validation |
| `POST /api/deep/train/` | `deep` configuration and current `dataset_id`/`revision_id`; CNN instead requires `deep.image_id` from a validated collection |
| `GET /api/deep/<id>/checkpoint/` | Owner-only best `.pt` checkpoint; available after first completed epoch, including interrupted runs |
| `GET /api/deep/<id>/output/` | Owner-only completed neural test outputs as CSV |
| `GET /api/recipe/` | Applied recipe as JSON |

### Request examples

These examples show request bodies, not authentication bypasses. Substitute the IDs returned by **your own** current `/api/state/` response. Include the immutable `revision_id` as well as its display number when using the general jobs endpoint.

**Drop an explicitly selected column:**

```json
{
  "kind": "clean",
  "dataset_id": "<dataset UUID from state>",
  "revision": 2,
  "revision_id": 17,
  "operation": {
    "action": "drop_selected_columns",
    "selection": {"columns": ["sample_id"]}
  }
}
```

**Run a correlation report:**

```json
{
  "kind": "analysis",
  "dataset_id": "<dataset UUID from state>",
  "revision": 2,
  "revision_id": 17,
  "analysis": {
    "type": "correlation",
    "columns": ["concentration", "temperature"],
    "method": "spearman"
  }
}
```

**Train Ridge with a small grid search:**

```json
{
  "kind": "model",
  "dataset_id": "<dataset UUID from state>",
  "revision": 2,
  "revision_id": 17,
  "model": {
    "model": "ridge",
    "features": ["temperature", "season"],
    "target": "concentration",
    "params": {"alpha": 1},
    "test_size": 0.2,
    "seed": 42,
    "scaling": "standard",
    "search": {"method": "grid", "ranges": {"alpha": [0.1, 1, 10]}, "folds": 3}
  }
}
```

**Train an MLP via `POST /api/deep/train/`:**

```json
{
  "dataset_id": "<dataset UUID from state>",
  "revision_id": 17,
  "deep": {
    "architecture": "mlp",
    "task": "regression",
    "features": ["temperature"],
    "target": "concentration",
    "hidden_layers": [64, 32],
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 30,
    "early_stopping": true,
    "patience": 5,
    "validation_size": 0.2,
    "test_size": 0.2,
    "device": "auto"
  }
}
```

For CNN training, pass `{"deep":{"architecture":"cnn","image_id":"<your validated image-job UUID>"}}` plus optional neural settings. It does not require an active tabular dataset. Folder uploads send repeated `images` multipart fields and a JSON `paths` array in matching order; ZIP uploads send one `archive` field.

### Responses and errors

| Status / shape | Meaning and next action |
| --- | --- |
| `202` with `id`, `kind`, `status`, `progress`, `message`, `result`, `created_at` | Accepted job. Poll `GET /api/jobs/<id>/`; inspect terminal status. |
| `200` on reads | State, preview, catalog or authorized artifact response. Artifact endpoints stream files rather than wrapping bytes in JSON. |
| `400` | Invalid options, stale revision, wrong selection, unavailable artifact or another active workspace job. Read the error message before retrying. |
| `403` | Authentication/CSRF failure under session authentication. Sign in and supply a valid CSRF token. |
| `404` | Missing resource or a resource outside the current owner's workspace. |
| `429` | Upload/job throttle exceeded. Wait for the limit window or review deployment rate policy. |
| `503` at submission | Broker/cache unavailable, or an immediately failed task in eager-test execution. The normal async worker failure is reported later in Job status. |
| Job `failed` | Request was accepted but work could not finish. `message` explains supported validation/time-limit failures; inspect worker logs for unexpected errors. |

The app also exposes `/`, `/accounts/login/`, `/accounts/logout/`, `/accounts/register/`, `/workspace-status/` and `/admin/`. Password-reset email delivery and production admin access require deployment configuration; the project does not provide a turnkey email service. The launcher alone adds `/_local/health/` to identify its own running instance.

## Source code map

| Path | Responsibility |
| --- | --- |
| `manage.py` | Django command entry point: checks, migrations, tests, runserver and administrative commands. |
| `project/settings.py` | Environment configuration, apps/middleware, PostgreSQL defaults, authentication, limits, Celery and DRF. |
| `project/local_settings.py` | Persistent single-computer SQLite, memory broker/cache and separate local cookies. |
| `project/test_settings.py` | Isolated test database, test secret/hashers, memory broker and eager task behavior. |
| `project/urls.py`, `wsgi.py`, `celery.py` | HTTP routing, WSGI application and Celery application initialization. |
| `workspace/models.py`, `migrations/`, `admin.py` | Workspace/upload/dataset/revision/job records, schema and admin registrations. |
| `workspace/views.py` | Authenticated page rendering, registration and status fragment. |
| `workspace/api.py`, `urls.py` | APIs, ownership checks, job submission, preview, recipes, classical model artifacts and prediction upload. |
| `workspace/uploads.py` | Aggregate upload-byte cap before unbounded multipart writes. |
| `workspace/ingestion.py` | Safe source validation, parsing, schema normalization, table/sheet discovery and protected remote fetch. |
| `workspace/cleaning.py` | Selection resolution, deterministic transformations and duplicate/outlier flags. |
| `workspace/storage.py` | Private paths, profiles, Parquet snapshot reads/writes and export formatting. |
| `workspace/tasks.py` | Atomic job claim, worker dispatch, revision commits, recipe staging, report/artifact persistence and error handling. |
| `workspace/analytics/common.py` | Shared report/table/Plotly builders, selection validation and JSON-safe numbers. |
| `workspace/analytics/correlation.py`, `regression.py`, `anova.py`, `plots.py` | Statistical method implementations and diagnostics. |
| `workspace/ml/catalog.py` | Whitelisted estimators and hyperparameter specs; source of the UI catalog. |
| `workspace/ml/preprocessing.py` | Feature schema, train-fitted preprocessing, new-row validation and design-size guard. |
| `workspace/ml/training.py`, `evaluation.py`, `prediction.py` | Training/search, task-specific reports and fitted-pipeline scoring. |
| `workspace/deep/config.py`, `networks.py`, `data.py` | Neural bounds/defaults, six network builders, splits, train-only scalers and sequence windows. |
| `workspace/deep/images.py` | Bounded image-archive validation, decoding, normalization and deduplication. |
| `workspace/deep/training.py` | PyTorch optimization, device selection, epoch history, early stopping, best checkpoints and held-out outputs. |
| `workspace/deep/jobs.py`, `api.py` | Owner-scoped image/deep jobs, persistent progress and private downloads. |
| `templates/base.html`, `registration/` | Common HTML shell and login/registration pages. |
| `templates/workspace/index.html` | Data explorer, cleaning controls, import/export/history and workflow shell. |
| `templates/workspace/statistics.html`, `modeling.html`, `deep.html`, `report.html` | Per-workflow controls and shared report rendering. |
| `static/js/workspace.js` | Alpine workspace state, table interactions, uploads, cleaning and job polling. |
| `static/js/analysis.js`, `modeling.js`, `deep.js` | Analysis/report controls, catalog/search/prediction controls, neural/image/live-chart controls. |
| `assets/input.css`, `tailwind.config.js`, `scripts/vendor.mjs` | Frontend build sources and local third-party asset copying. |
| `static/css/`, `static/vendor/` | Built, checked-in CSS and pinned browser assets. |
| `start.bat`, `scripts/start_local.py`, `scripts/init_env.py` | Windows launch wrapper, persistent local runtime and secret initialization. |
| `scripts/smoke_server.py` | Isolated real-worker UI preview on port 8787. |
| `Dockerfile`, `compose.yaml`, `compose.gpu.yaml` | Development containers, service coordination, optional deep-worker GPU reservation. |
| `workspace/tests/test_platform.py` | Ingestion, cleaning, storage/API protections and preparation workflows. |
| `workspace/tests/test_analytics.py`, `test_ml.py`, `test_deep.py` | Statistical correctness, classical models/persistence and neural/image workflows. |
| `docs/media/`, `scripts/build_readme_media.py` | Actual UI captures and reproducible screenshot-sequence GIF assembly. |
| `docs/model-parameters.md`, `scripts/build_model_reference.py` | Generated complete classical-model parameter reference and its generator. |

### Historical research material

The root also contains `Checking of data to model.ipynb`, `modelfunction.ipynb`, `MySQL to csv.ipynb`, `sqltocsv.py`, `updated_sqltocsvcodes.py`, a Word model-function document and historical CSVs. They are preserved research artifacts, are not imported by the Django application, and may rely on their own historical dependencies or external data. This guide's runtime instructions apply to the web platform, not automatically to those notebooks/scripts. The Docker build excludes notebooks and Word documents.

### Files deliberately excluded from Git

`.gitignore` excludes `venv/`, `.venv/`, `env/`, `.env`, databases, `media/`, `staticfiles/`, `.tmp/`, `node_modules/`, Python bytecode, caches, test coverage and logs. Documentation screenshots are deliberately checked in under `docs/media/` and contain only demo content. Never capture production datasets or account/session details into that directory.

## Validation

```bash
python manage.py test workspace.tests --settings=project.test_settings
python manage.py check --settings=project.test_settings
python manage.py makemigrations --check --dry-run --settings=project.test_settings
pip check
node --check static/js/workspace.js
node --check static/js/analysis.js
node --check static/js/modeling.js
node --check static/js/deep.js
npm run build
```

The 66 automated tests use temporary media and isolated SQLite. They cover parsers, rejected SQL/XML/URL inputs, selection, drop/undo/replay, cleaning, atomic recipes, exports, CSRF, ownership, broker failures, throttles, and previews. Statistical tests compare results with SciPy/statsmodels/Pingouin and exercise all analysis/plot families. ML tests fit and serialize all 28 estimators, check training-only and fold-local preprocessing, held-out metric calculations, grid/random search, cutoff counts, new categories, artifact ownership, prediction uploads, and asynchronous enqueueing. Deep tests train all six architectures, restore best checkpoints and compare outputs, exercise all optimizers/activations, verify chronological window boundaries and training-only scalers, test early stopping/interrupted checkpoints, validate image archives/folders, and check dedicated queue routing and ownership. CPU training is tested; CUDA hardware is unavailable here.

The optional browser-test harness uses a **real Celery worker** with an in-memory broker and isolated SQLite files:

```bash
python scripts/smoke_server.py
```

Open [the isolated test app](http://127.0.0.1:8787). It binds only to loopback and stores test data under ignored `.tmp/ui-smoke/`. It uses test settings and is not a deployment entry point. The default `manage.py runserver` setup uses PostgreSQL and Redis; `start.bat` uses the separate persistent local mode. Eager tasks are confined to automated tests.

### Working on the project

1. Keep dependencies in the project venv and use the pinned Python version.
2. Make backend changes in the responsible module; keep expensive work inside workers.
3. For a new estimator, update the catalog, constructor and relevant task-specific evaluation/prediction behavior. Regenerate the model reference.
4. For new operations, define serializable parameters and verify selection, undo/replay and failure behavior.
5. For a new neural architecture, update validated configuration, network reconstruction, preprocessing and checkpoint round-trip checks together.
6. Rebuild CSS/vendor assets after frontend/template changes. Review the browser at desktop and narrow widths.
7. Run tests appropriate to the change; run the full suite before merging a broad implementation change.
8. Document new limits, data assumptions and artifact compatibility, rather than implying every dataset is supported.

Phase 4 verification passed **66 automated tests**, exercised actual MLP/CNN jobs and epoch polling in the browser, and checked launcher setup and dependency consistency. Neural execution was verified on CPU. CUDA and Docker GPU passthrough require hardware/environment testing on the deployment target. This README-only update does not claim a new GPU validation.

## Operations

- Jobs stuck **queued**: start the worker and check that Django/worker share database, Redis, and media configuration.
- Broker unavailable: submission reports the error and releases the workspace lock.
- A forcibly killed worker can leave a **running** job. Confirm its worker has stopped, inspect the current dataset revision, then mark the interrupted job failed in Django admin before resubmitting. Never release a job while its worker may still be writing.
- Deployment needs a WSGI/ASGI server, TLS, `DEBUG=False`, fresh credentials, shared private storage, reverse-proxy upload caps, retention, and process/container resource limits. Run Django's deployment checks with that configuration. [Django deployment checklist](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/).

### Troubleshooting by symptom

| Symptom | Likely cause / resolution |
| --- | --- |
| Double-click closes with an error / version mismatch | Read the retained console message. Install Python 3.12.10 and ensure the project's venv was created with it. |
| First startup takes a long time | Missing pinned packages are being installed, including the large PyTorch wheel. Internet is needed for initial installation. |
| Port already in use | Close the other server if appropriate, or use `start.bat --port 8001`. The launcher will not attach to an unrelated app. |
| My account/data disappeared between 8000 and 8787 | Those modes use different metadata/media stores. Return to the launcher mode used to create the data. |
| Job stays queued | Check the broker and worker. Ordinary jobs need the `celery` queue; neural/image jobs need the `deep` queue. Check database/media settings match. |
| New action says a job is already running | Only one active job is permitted per workspace. Wait for completion, or diagnose a stopped worker before any manual recovery. |
| Workbook import shows choices instead of a table | Select worksheet(s); table-based sources need a table selection. This is the intended two-step import. |
| Remote import rejected | Check public visibility, exact host allowlist, HTTPS, direct CSV response and `gid` in the query rather than URL fragment. Download and upload a local file if needed. |
| Cleaning asks to clear row/cell selection | The chosen operation changes complete columns. Clear selection, then choose the intended columns again. |
| Fill leaves gaps | No inferable values, endpoint interpolation gaps or empty groups can remain. Inspect the selected columns and choose a justified constant or other method. |
| Report says earlier/different revision | Data changed, an Undo branch was replaced, or the report refers to another dataset. Re-run using the current revision if desired. |
| A model rejects dates or text | Convert datetime features meaningfully; deep tabular inputs require numeric features. Classical ML supports categorical features through its pipeline. |
| Search has no valid candidate | Check fold counts, class counts, neighbors/components and parameter bounds. Reduce the search or add appropriate observations. |
| CNN validation rejects an archive | Check class-folder structure, supported formats, extra files, duplicate/conflicting labels and distinct images per class. |
| Sequence split is too small | Each raw partition needs at least window length + 2 rows. Reduce the window or add rows while preserving chronological order. |
| CUDA required fails | Verify the worker's Python environment has the compatible CUDA wheel and can access a supported device. Choose CPU/Auto otherwise. |
| Training diverges | Lower the learning rate, inspect extreme values and simplify the network. The best previously completed checkpoint can remain available. |
| Download unavailable | The job may not be ready, may belong to another user, or its private artifact may have been moved/deleted. Inspect the job and storage. |
| Charts/styles look outdated | Rebuild frontend assets and reload the page. Restart the launcher after backend/template updates. |

### Back up and restore coherently

For the local launcher, stop it cleanly before copying the complete `media/local/` directory. Back up the metadata SQLite database and its corresponding private dataset/artifact directory together. Keep `.env` separately and securely so the installation retains its intended settings. Recreating `venv/` from pinned requirements is preferable to treating the environment as the only backup.

For PostgreSQL/Compose, take a database backup and a consistent backup of the private-media volume. A metadata-only backup leaves file paths pointing at missing snapshots; a media-only backup loses accounts, ownership, job state and revision relationships. Restore under matching settings, verify path availability/permissions, then check representative dataset previews and artifact downloads before accepting new work.

The application has no automatic file-retention scheduler or disk quota manager. Clear/replacement and undo-branch changes leave historical files behind. A deployment cleanup process must account for live jobs, revisions, uploads, reports and model/checkpoint relationships before removing files. Do not delete the whole media directory to recover a stuck job.

### Deployment readiness

The supplied server commands and Compose stack are development entry points. A hosted deployment must provide a production WSGI/ASGI server, TLS termination, correct host/proxy configuration, secure credentials, shared private storage, ingress caps, backups and worker resource limits. Secure-cookie behavior depends on `DEBUG`; configure it deliberately. Run `python manage.py check --deploy` against actual deployment settings. Never expose the smoke server, test hashers or local memory-broker setup as a multi-user production service.

## FAQ and current boundaries

**Does the browser keep my full dataset in memory?** No. It receives a bounded preview plus cached profile/report data. Workers still load full applicable frames/designs into their own memory, so this is not an out-of-core training platform.

**Can I keep multiple tables active or join them?** A workspace has one active tabular dataset. Multi-sheet import can combine worksheets, and image collections are separate. A general multi-dataset join/relationship editor is not implemented.

**Does filtering the preview train on only visible rows?** No. Search/filter/sort are inspection tools. Training uses the selected columns of the complete active revision with its own documented split/filter rules.

**Can I upload a trained pickle or neural model?** No. Classical prediction uses a previously saved server-owned model; neural checkpoints can be downloaded and used from Python. Uploaded arbitrary model deserialization is intentionally absent.

**Can I resume an interrupted neural run at its next epoch?** No. The retained best checkpoint supports inference/reuse, but optimizer state and training-resume scheduling are not saved.

**Can I cancel a running job from the UI?** There is no cancel control in this release. Jobs have input/resource bounds; deployment-level process handling requires care to preserve consistent job state.

**Are there built-in deployments for serving a model as a public prediction API?** No. Private artifact downloads and the classical new-file scoring workflow are implemented; public model serving and scheduled retraining are not.

**Are classification scores guaranteed to improve with more epochs or a larger model?** No. The training curves and held-out metrics expose what happened; more training can overfit. Very small demo splits have unstable scores and are for workflow demonstration.

**Will a GPU always be used?** Only if the selected worker has a compatible CUDA PyTorch build and accessible device. Auto transparently uses CPU otherwise; CPU fallback is recorded in the report.

**Does this include experiment tracking across teams, role-based sharing or real-time collaboration?** No. It implements per-user ownership, recent saved runs and reproducible artifacts. Team permissions, registry/version promotion, distributed training and collaboration are outside the current scope.

**Where are data files served from?** They are not exposed as a public media tree. Authenticated endpoints check ownership before streaming approved artifacts.

## Documentation media

The README includes eight feature walkthroughs plus an opening platform tour—nine looping GIFs—and a gallery of genuine PNG captures under [docs/media](docs/media/). See [the media guide](docs/media/README.md) for each animation's exact frame order and capture notes. Each feature section includes a still image or a direct link to the source captures, so readers can inspect a static view when motion is distracting. Open any PNG at full size for small labels.

To rebuild the GIFs from the committed screenshot frames and refresh the parameter catalog:

```bash
python scripts/build_readme_media.py
python scripts/build_model_reference.py
```

These utilities run inside the existing project venv. GIFs use deliberately slow, discrete screen changes and loop without JavaScript or remote image services. They are screen-state walkthroughs, not recordings of exact user interaction timing. To update an obsolete frame, capture the same workflow in the isolated smoke app with synthetic data, replace the corresponding PNG, rerun the media builder, and inspect the result before committing.

<details>
<summary>Static screenshot gallery — open for motion-free feature views</summary>

| Feature | Screenshot |
| --- | --- |
| Data explorer and profile | [Open screenshot](docs/media/01-data-explorer.png) |
| Import/parser options | [Open screenshot](docs/media/02-import-options.png) |
| Operation history | [Open screenshot](docs/media/04-history.png) |
| Missing values | [Open screenshot](docs/media/06-missing-values.png) |
| Selected numeric columns | [Open screenshot](docs/media/07-cleaning-selection.png) |
| Cleaned revision | [Open screenshot](docs/media/08-cleaned-data.png) |
| Drop-column operation | [Open screenshot](docs/media/09-drop-column.png) |
| Correlation report | [Open screenshot](docs/media/10-correlation.png) |
| Correlation matrix | [Open screenshot](docs/media/11-correlation-matrix.png) |
| Regression diagnostics | [Open screenshot](docs/media/12-regression-diagnostics.png) |
| Classical model report | [Open screenshot](docs/media/13-model-report.png) |
| Model downloads/prediction | [Open screenshot](docs/media/14-model-outputs.png) |
| Classification cutoff | [Open screenshot](docs/media/16b-cutoff-half.png) |
| CNN configuration | [Open screenshot](docs/media/17-deep-configuration.png) |
| Neural results | [Open screenshot](docs/media/18-deep-report.png) |
| Live neural training | [Open screenshot](docs/media/20-live-training-b.png) |
| Neural loss/accuracy curves | [Open screenshot](docs/media/21-deep-curves.png) |
| Data export | [Open screenshot](docs/media/22-export-csv.png) |

</details>

## Roadmap

1. **Delivered:** ingestion, preview, cleaning, export, authentication, background jobs, setup.
2. **Delivered:** column-selected statistics, correlation, regression diagnostics, ANOVA, diagnostic and EDA plots, persistent reports.
3. **Delivered:** classical ML, grid/random tuning, held-out evaluation, clustering/reduction, model downloads, prediction.
4. **Delivered:** six PyTorch architectures, separate image uploads, CUDA-capable worker queue with CPU fallback, live epoch progress, early stopping, held-out evaluation, and checkpoints.

`requirements-later-phases.txt` is a compatibility entry point that includes the complete `requirements.txt`.
