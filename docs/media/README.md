# UI screenshots and animated walkthroughs

[Return to the project guide](../../README.md#documentation-media)

## Provenance

Captured from the running Django application in its isolated `scripts/smoke_server.py` demo on September 16, 2026. The visible account is the local QA account `studio_qa`. Tables contain the built-in synthetic water-quality sample; the CNN collection contains generated test images. No production datasets, secrets, session cookies or personally identifying records are shown.

PNG files are actual browser screenshots. Their UI text, metrics and charts have not been fabricated or retouched. Early captures use the browser's current responsive size; later captures use a consistent desktop canvas for GIF assembly. GIFs preserve the screenshot pixels and pad differing canvas sizes as necessary. Palette quantization is part of GIF encoding. The original PNGs are the static, full-color alternative.

## Walkthroughs

| GIF | Exact PNG sequence | What it explains |
| --- | --- | --- |
| [Platform tour](platform-tour.gif) | 01 → 10 → 13 → 18 | Data preparation, statistics, classical modeling and neural results. |
| [Import options](import-options.gif) | 02 → 03 | Parser dialog with automatic choices, then explicit semicolon/UTF-8 settings. No dataset replacement is submitted. |
| [Preparation](preparation.gif) | 06 → 07 → 08 | Five missing cells, selected numeric columns, then the existing saved mean-fill revision restored with Redo. |
| [History](history.gif) | 04 → 05 | Undo moves the current revision from 2 to 1; subsequent steps remain available for Redo. |
| [Statistics](statistics.gif) | 10 → 11 → 12 | Current correlation setup/report, matrix detail and a saved regression diagnostics view. |
| [Modeling](modeling.gif) | 13 → 14 → 16b | Saved Ridge model, downloads/new-file prediction controls, and a separate logistic classifier's threshold panel. |
| [Classification cutoff](classification-cutoff.gif) | 15 → 16b → 16 | Same fitted classifier at cutoffs 0, 0.5 and 1; confusion counts change without retraining. |
| [Deep learning](deep-learning.gif) | 17 → 19 → 20 → 18 → 21 | CNN controls, an actual run at epochs 7 and 47, final best-checkpoint metrics and curves. The configuration frame precedes the longer capture run; read the saved report for its actual settings. |
| [Export formats](export-formats.gif) | 22 → 23 → 24 | CSV, JSON and Parquet choices; choosing a format alone does not export or alter data. |

The guide has **eight feature walkthroughs plus the opening platform tour**. Each GIF advances every 2.6 seconds and loops. These are discrete captured screen states, not real-time recordings or speed comparisons. Demo holdouts contain very few rows and the generated image classes have no meaningful predictive signal; displayed model scores are not performance claims.

## Rebuilding

From the repository root, using the existing venv:

```bash
python scripts/build_readme_media.py
```

The builder reads the PNGs, preserves their pixels on a common canvas and writes the nine GIF files. To refresh the images, use the isolated demo with synthetic data, navigate through actual UI controls, capture the corresponding view and replace its PNG. Preserve the source filenames or update `SEQUENCES` and README links together. Verify the first, intermediate and final frames, not only the GIF file header.

## Static gallery

- **Explorer and preparation:** [01](01-data-explorer.png), [06](06-missing-values.png), [07](07-cleaning-selection.png), [08](08-cleaned-data.png), [09](09-drop-column.png).
- **Import:** [02](02-import-options.png), [03](03-import-configured.png).
- **History:** [04](04-history.png), [05](05-history-undo.png).
- **Statistics:** [10](10-correlation.png), [11](11-correlation-matrix.png), [12](12-regression-diagnostics.png).
- **Classical models:** [13](13-model-report.png), [14](14-model-outputs.png), [15](15-cutoff-zero.png), [16b](16b-cutoff-half.png), [16](16-cutoff-one.png).
- **Neural networks:** [17](17-deep-configuration.png), [18](18-deep-report.png), [19](19-live-training-a.png), [20](20-live-training-b.png), [21](21-deep-curves.png).
- **Exports:** [22](22-export-csv.png), [23](23-export-json.png), [24](24-export-parquet.png).
