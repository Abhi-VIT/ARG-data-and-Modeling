# ARG Data Studio

A Django workspace for preparing, analyzing, and modeling data without writing Python. **Phases 1–4 are implemented:** ingestion, cleaning, export, statistics, classical machine learning, PyTorch deep learning, and interactive plots.

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

## Workspace guide

1. **Import:** choose a file or URL and parsing options. Excel and SQL/SQLite show available sheets/tables before import. Combining sheets unions columns and adds `_source_sheet`.
2. **Explore:** search all columns, apply contains-filters, and click ↕ to sort. Hover numeric headers for histograms. Only one page of rows is mounted; wide datasets scroll horizontally.
3. **Select:** click a cell for its editor, a column name/checkbox for a column, or a row number for a row. Shift-click a second cell for a rectangular range. Row numbers identify positions in the current revision, even after sorting/filtering. A range uses these numbers, not filtered display order. No selection means the whole dataset.
4. **Clean:** choose an operation and method. To drop columns, explicitly select their header checkboxes, choose **Drop selected columns**, and apply. Undo restores them, and recipes replay the drop. Keep at least one column. Structural column changes require all rows. Numeric methods reject incompatible columns with a clear message. Outlier detection flags rows in amber; review before choosing **Remove flagged rows**.
5. **History:** undo/redo or download a recipe. An edit after undo replaces the redo branch. Replay requires exactly matching ordered input column names; each recorded operation creates its own revision.
6. **Export:** a worker prepares the current revision and starts its download. Reopen Export data to download again.
7. **Statistics:** choose an analysis, explicitly check columns or **Select all applicable**, assign roles, then **Run analysis**. Numeric analyses list excluded nonnumeric columns. Reports retain their source revision; cleaning does not overwrite them. Use the saved-report menu to reopen recent results, download report JSON, or download individual charts as PNG.
8. **Models:** choose a learning task and model, select a target for supervised tasks, and check features or **Select all except target**. Adjust the split, scaling, seed, and hyperparameters, then **Train model**. Saved model runs can be reopened from the report menu.
9. **Deep learning:** choose an architecture, select numeric features and a target (or upload a CNN image collection), configure training, then **Train neural network**. The live panel shows epoch loss/accuracy and the best completed checkpoint. Saved runs retain their curves, metrics, and outputs.

### Phase 2 analysis guide

- **Correlation:** Auto chooses a compatible method per pair; override with a named method when appropriate. Pearson measures linear association, Spearman/Kendall measure rank association, point-biserial handles binary/numeric pairs, Phi handles two binary columns, and Cramér’s V handles categorical pairs. Reports document binary coding and incompatible pairs. Pairwise complete observations are used; p-values are unadjusted for multiple comparisons.
- **Regression:** select the numeric target and predictors. Custom models add degree 2/3 polynomials and pairwise interactions through the formula builder. Formulas are generated from internal aliases, so uploaded column names cannot execute code. Forward, backward, and bidirectional selection support AIC, BIC, or nested F-test p-values and preserve term hierarchy. Final inference does not account for selection. All candidates use the same complete rows.
- **Diagnostics:** coefficient tables include standard errors, t-tests, p-values, and confidence intervals. Reports include R², adjusted R², F-tests, AIC/BIC, Shapiro–Wilk, Anderson–Darling critical values, D’Agostino–Pearson, Breusch–Pagan, White, Durbin–Watson, Ljung–Box, and configurable VIF flags. The KS normality check uses [Lilliefors calibration](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.lilliefors.html) because distribution parameters are estimated. Autocorrelation follows dataset row order, not display sorting. Undefined or inapplicable statistics show a dash with explanatory notes.
- **ANOVA:** ordinary one/two-way models assume independent groups and equal variances. Two-way includes interaction and uses type II sums of squares. Welch permits unequal variances. Repeated measures accepts one within-subject factor in long format, requires one observation per subject/condition, and excludes incomplete subjects. Reports show [sphericity and Greenhouse–Geisser correction](https://pingouin-stats.org/generated/pingouin.rm_anova.html). Tukey is available for ordinary ANOVA only; two-way Tukey pools the other factor and is not a simple-effects test.
- **Plots:** histogram, box, violin, scatter, pair, Pearson heatmap, categorical counts, and line charts. Line charts sort by X. Regression includes residuals vs fitted, normal Q–Q, scale–location, Cook’s distance, and residuals vs leverage. Large point/distribution plots are sampled and explicitly labeled; statistical tables use all applicable observations.

### Phase 3 modeling guide

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

### Phase 4 deep-learning guide

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
| Reports | Latest 30 successful reports appear in the menu. Older reports remain available by their private job URLs; JSON includes parameters, tables, figures, warnings, and source revision. |
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

```text
Django templates + Alpine / HTMX
        │ authenticated, CSRF-protected DRF requests
        ▼
PostgreSQL: workspace → dataset → revisions + jobs
        │ enqueue                         ▲ poll status
        ▼                                 │
Redis → Celery workers ────────────────────┘
        │ parse / clean / profile / analyze / export
        ▼
Private disk: sources + Parquet snapshots + exports + reports
```

- `project/`: settings, routing, and Celery configuration.
- `workspace/ingestion.py`: validators, format parsers, protected URL fetching.
- `workspace/cleaning.py`: deterministic cleaning operations.
- `workspace/analytics/`: statistical computations and Plotly report generation inside the worker.
- `workspace/ml/`: estimator catalog, validated preprocessing, supervised/search and unsupervised training, evaluation, and prediction.
- `workspace/deep/`: PyTorch architectures, training-only preprocessing, safe image ingestion, epoch/checkpoint persistence, and private neural APIs.
- `workspace/tasks.py`: worker orchestration and revision commits.
- `workspace/api.py`: authenticated API, job submission, bounded preview, downloads.
- `workspace/storage.py`: Parquet storage, profiles, and exports.
- `templates/`, `assets/`, `static/`: frontend source and built assets.
- Original notebooks, CSVs, and MySQL scripts are preserved as research material. They are not used by the web application and may need their own historical dependencies.

### API

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

### Operations

- Jobs stuck **queued**: start the worker and check that Django/worker share database, Redis, and media configuration.
- Broker unavailable: submission reports the error and releases the workspace lock.
- A forcibly killed worker can leave a **running** job. Confirm its worker has stopped, inspect the current dataset revision, then mark the interrupted job failed in Django admin before resubmitting. Never release a job while its worker may still be writing.
- Deployment needs a WSGI/ASGI server, TLS, `DEBUG=False`, fresh credentials, shared private storage, reverse-proxy upload caps, retention, and process/container resource limits. Run Django's deployment checks with that configuration. [Django deployment checklist](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/).

## Roadmap

1. **Delivered:** ingestion, preview, cleaning, export, authentication, background jobs, setup.
2. **Delivered:** column-selected statistics, correlation, regression diagnostics, ANOVA, diagnostic and EDA plots, persistent reports.
3. **Delivered:** classical ML, grid/random tuning, held-out evaluation, clustering/reduction, model downloads, prediction.
4. **Delivered:** six PyTorch architectures, separate image uploads, CUDA-capable worker queue with CPU fallback, live epoch progress, early stopping, held-out evaluation, and checkpoints.

`requirements-later-phases.txt` is a compatibility entry point that includes the complete `requirements.txt`.
