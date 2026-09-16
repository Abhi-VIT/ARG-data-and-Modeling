"""Generate model documentation from the catalog that drives the UI."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from workspace.ml.catalog import CATALOG  # noqa: E402

MEANINGS = {
    'fit_intercept': 'Fit a constant intercept term.',
    'alpha': 'Penalty or smoothing strength; interpretation depends on the model.',
    'l1_ratio': 'ElasticNet mixture: 0 is L2, 1 is L1.',
    'degree': 'Maximum degree of the polynomial feature expansion.',
    'C': 'Inverse regularization strength for the margin/logistic objective.',
    'epsilon': 'SVR loss-insensitive margin around the target.',
    'kernel': 'Kernel used to represent feature relationships.',
    'gamma': 'Kernel coefficient convention.',
    'max_depth': 'Maximum depth of an individual tree.',
    'min_samples_split': 'Minimum observations needed to split a tree node.',
    'min_samples_leaf': 'Minimum observations remaining in a tree leaf.',
    'n_estimators': 'Number of trees or boosting stages.',
    'max_features': 'Feature subset considered at a tree split.',
    'learning_rate': 'Boosting step size, or t-SNE optimization rate.',
    'subsample': 'Fraction of observations used per boosting iteration.',
    'colsample_bytree': 'Fraction of features sampled for each XGBoost tree.',
    'num_leaves': 'Maximum leaves per LightGBM tree.',
    'min_child_samples': 'Minimum observations in a LightGBM leaf.',
    'n_neighbors': 'Neighborhood size for KNN or UMAP.',
    'weights': 'Uniform or distance-weighted neighbor contributions.',
    'p': 'Minkowski distance power: 1 Manhattan, 2 Euclidean.',
    'class_weight': 'No class weighting or weighting balanced by class frequencies.',
    'max_iter': 'Maximum optimization iterations.',
    'var_smoothing': 'Added variance stabilization for Gaussian naive Bayes.',
    'fit_prior': 'Learn class priors from training frequencies.',
    'n_clusters': 'Requested cluster count.',
    'n_init': 'Number of K-Means initializations.',
    'linkage': 'Criterion used when merging hierarchical clusters.',
    'eps': 'DBSCAN neighborhood radius in preprocessed feature space.',
    'min_samples': 'Minimum neighborhood support for a DBSCAN core point.',
    'n_components': 'Number of retained projection/embedding coordinates.',
    'whiten': 'Rescale PCA components to unit variance.',
    'perplexity': 't-SNE effective neighborhood scale; must also suit row count.',
    'min_dist': 'UMAP minimum separation parameter in the embedding.',
}


def literal(value):
    return '`' + json.dumps(value, ensure_ascii=False) + '`'


def build():
    lines = ['# Complete model parameter reference', '',
             '[Return to the project guide](../README.md#phase-3-modeling-guide)', '',
             'Generated from `workspace/ml/catalog.py` with `python scripts/build_model_reference.py`. '
             'These are the exact exposed controls, not every parameter supported by the upstream libraries. '
             'Sample-count and design-size guards can impose tighter effective bounds.', '',
             f'**{len(CATALOG)} estimator entries.** Regression and classification variants have separate keys.', '',
             '| Task | Entries |', '| --- | --- |']
    for task in ['regression', 'classification', 'clustering', 'reduction']:
        lines.append(f'| {task} | {sum(item["task"] == task for item in CATALOG.values())} |')
    for task in ['regression', 'classification', 'clustering', 'reduction']:
        lines.extend(['', f'## {task.capitalize()}', ''])
        for key, model in CATALOG.items():
            if model['task'] != task:
                continue
            lines.extend([f'### {model["name"]} — `{key}`', '',
                          '| Parameter | Type | Default | Allowed | Meaning |',
                          '| --- | --- | --- | --- | --- |'])
            for name, spec in model['params'].items():
                allowed = ', '.join(map(literal, spec['values'])) if spec['type'] == 'choice' else f'{literal(spec["min"])}–{literal(spec["max"])}'
                lines.append(f'| `{name}` | {spec["type"]} | {literal(spec["default"])} | {allowed} | {MEANINGS[name]} |')
    lines.extend(['', '## Shared training controls', '',
                  '| Control | Default | Constraints |', '| --- | --- | --- |',
                  '| Features | None selected | 1–50 distinct feature names; numeric/category support; convert dates first. |',
                  '| Target | None selected | Required for supervised tasks; separate from features. |',
                  '| Test fraction | 0.2 | 0.1–0.5; supervised tasks only. |',
                  '| Stratification | Enabled | Classification only; sample counts must permit the split. |',
                  '| Seed | 42 | Integer 0–2,147,483,647. |',
                  '| Scaling | standard | standard / minmax / robust / none. Multinomial NB forces minmax with clipping. |',
                  '| Search method | none | none / grid / random; supervised only. |',
                  '| CV folds | 3 | 2–5; fitted preprocessing is refit inside each fold. |',
                  '| Random-search iterations | 10 | 1–20; each searched parameter has 1–10 listed candidates. |',
                  '', 'See the README for data-size limits, scoring, pipeline persistence and supported new-row prediction.', ''])
    destination = ROOT / 'docs' / 'model-parameters.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Generated {destination.relative_to(ROOT)} for {len(CATALOG)} models.')


if __name__ == '__main__':
    build()
