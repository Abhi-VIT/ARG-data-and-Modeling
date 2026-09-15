import json
import warnings
from .common import clean_json
from .correlation import correlation
from .regression import regression
from .anova import anova
from .plots import descriptive, eda
from ..ingestion import DataError

ANALYSES = {'descriptive': descriptive, 'correlation': correlation, 'regression': regression, 'anova': anova, 'eda': eda}


def analyze(frame, options, progress=lambda percent, message: None):
    if not isinstance(options, dict) or options.get('type') not in ANALYSES:
        raise DataError('Choose a supported statistics analysis.')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = ANALYSES[options['type']](frame, options, progress)
    messages = list(dict.fromkeys(str(w.message)[:300] for w in captured))
    result['warnings'].extend(messages[:12])
    result['parameters'] = options
    result = clean_json(result)
    # Strict JSON guards against undefined test statistics leaking as NaN/Infinity.
    json.dumps(result, allow_nan=False)
    return result
