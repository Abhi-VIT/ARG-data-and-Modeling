"""OLS reports from validated selections; user text is never evaluated as a formula."""
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox, lilliefors
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .common import report, selected, complete_rows, figure, table, number, diagnostic, PLOT_ROWS
from ..ingestion import DataError


def build_terms(predictors, options):
    aliases = {column: f'x{i}' for i, column in enumerate(predictors)}
    terms = {alias: {'label': column, 'parents': []} for column, alias in aliases.items()}
    polynomials = options.get('polynomials', {})
    interactions = options.get('interactions', [])
    if not isinstance(polynomials, dict) or not isinstance(interactions, list):
        raise DataError('Use the polynomial and interaction controls to build the model.')
    if options.get('model', 'multiple') != 'custom' and (polynomials or interactions):
        raise DataError('Choose Custom model to add interactions or polynomial terms.')
    for column, degree in polynomials.items():
        if column not in aliases or type(degree) is not int or degree not in (2, 3):
            raise DataError('Polynomial terms require a selected numeric predictor and degree 2 or 3.')
        alias = aliases[column]
        previous = alias
        for power in range(2, degree + 1):
            expression = f'I({alias} ** {power})'
            terms[expression] = {'label': f'{column}^{power}', 'parents': [previous]}
            previous = expression
    for pair in interactions:
        if not isinstance(pair, list) or len(pair) != 2 or any(c not in aliases for c in pair) or pair[0] == pair[1]:
            raise DataError('Each interaction needs two different selected numeric predictors.')
        a, b = sorted(pair, key=predictors.index)
        terms[f'{aliases[a]}:{aliases[b]}'] = {'label': f'{a} × {b}', 'parents': [aliases[a], aliases[b]]}
    if len(terms) > 60:
        raise DataError('Choose at most 60 regression terms.')
    return aliases, terms


def select_terms(fit, terms, options, result):
    direction = options.get('stepwise', 'none')
    if direction not in {'none', 'forward', 'backward', 'both'}:
        raise DataError('Choose no selection, forward, backward or bidirectional selection.')
    if direction == 'none':
        return list(terms)
    if len(terms) > 12:
        raise DataError('Stepwise selection supports at most 12 candidate terms.')
    criterion = options.get('criterion', 'aic')
    if criterion not in {'aic', 'bic', 'pvalue'}:
        raise DataError('Choose AIC, BIC or p-value selection.')
    enter, leave = float(options.get('p_enter', .05)), float(options.get('p_exit', .1))
    if not 0 < enter < leave < 1:
        raise DataError('Stepwise thresholds must satisfy 0 < entry p < removal p < 1.')
    included = list(terms) if direction == 'backward' else []
    trace, seen = [], set()
    for iteration in range(50):
        key = tuple(included)
        if key in seen:
            result['warnings'].append('Stepwise stopped at a repeated term set.')
            break
        seen.add(key)
        current = fit(included)
        candidates = []
        for term, spec in terms.items():
            if term not in included and direction in {'forward', 'both'} and set(spec['parents']) <= set(included):
                proposal = [t for t in terms if t in included or t == term]
                candidate = fit(proposal)
                value = candidate.compare_f_test(current)[1] if criterion == 'pvalue' else getattr(candidate, criterion)
                if np.isfinite(value) and (value < enter if criterion == 'pvalue' else value < getattr(current, criterion) - 1e-8):
                    candidates.append((value, 'add', term, proposal))
            if term in included and direction in {'backward', 'both'} and not any(term in terms[t]['parents'] for t in included):
                proposal = [t for t in included if t != term]
                candidate = fit(proposal)
                value = current.compare_f_test(candidate)[1] if criterion == 'pvalue' else getattr(candidate, criterion)
                if np.isfinite(value) and (value > leave if criterion == 'pvalue' else value < getattr(current, criterion) - 1e-8):
                    candidates.append((-value if criterion == 'pvalue' else value, 'remove', term, proposal))
        if not candidates:
            break
        score, action, term, included = min(candidates, key=lambda item: item[0])
        trace.append({'Step': iteration + 1, 'Action': action, 'Term': terms[term]['label'],
                      'Criterion': criterion, 'Value': abs(score) if criterion == 'pvalue' else score})
    result['tables'].append(table('Stepwise selection path', trace, ['Step', 'Action', 'Term', 'Criterion', 'Value']))
    result['notes'].append('Stepwise selection preserves lower-order terms. Selection is exploratory: final p-values and confidence intervals do not account for model selection.')
    return included


def regression(frame, options, progress):
    result = report('Regression diagnostics', 'regression')
    target = options.get('target')
    if target not in frame or not pd.api.types.is_numeric_dtype(frame[target]):
        raise DataError('Choose a numeric target column.')
    data = selected(frame, options, result, numeric_only=True, minimum=2)
    if target not in data:
        raise DataError('Include the target in your column selection.')
    predictors = [c for c in data if c != target]
    if len(predictors) > 20:
        raise DataError('Choose at most 20 numeric predictors.')
    model_kind = options.get('model', 'multiple')
    if model_kind not in {'simple', 'multiple', 'custom'}:
        raise DataError('Choose simple, multiple or custom regression.')
    if model_kind == 'simple' and len(predictors) != 1:
        raise DataError('Simple regression needs exactly one predictor plus the target.')
    data = complete_rows(data, result).astype(float)
    if len(data) < 5 or data[target].nunique() < 2:
        raise DataError('Regression needs at least 5 complete rows and a nonconstant target.')
    constants = [c for c in predictors if data[c].nunique() < 2]
    if constants:
        raise DataError('Remove constant predictors from the selection: ' + ', '.join(constants))
    aliases, terms = build_terms(predictors, options)
    if len(data) * (len(terms) + 1) > 20_000_000:
        raise DataError('Model design exceeds 20 million cells. Reduce rows or terms.')
    if len(data) <= len(terms) + 2:
        raise DataError('Need more complete rows than candidate terms plus two.')
    internal = pd.DataFrame({'response': data[target], **{alias: data[c] for c, alias in aliases.items()}})
    fitted = {}

    def fit(names):
        key = tuple(names)
        if key not in fitted:
            if len(fitted) >= 200:
                raise DataError('Stepwise exceeded 200 candidate fits. Reduce the candidate terms.')
            # Only generated aliases, powers 2/3, and validated operators reach patsy.
            formula = 'response ~ ' + (' + '.join(names) if names else '1')
            model = smf.ols(formula, internal, eval_env=-1).fit()
            if not np.isfinite(model.model.exog).all():
                raise DataError('Polynomial or interaction terms overflowed. Scale predictors first.')
            fitted[key] = model
            if len(fitted) % 10 == 0:
                progress(40, f'Comparing regression models ({len(fitted)} fits)')
        return fitted[key]

    chosen = select_terms(fit, terms, options, result)
    model = fit(chosen)
    progress(60, 'Computing residual diagnostics and influence')
    labels = {name: spec['label'] for name, spec in terms.items()}
    labels['Intercept'] = 'Intercept'
    result['formula'] = target + ' ~ ' + (' + '.join(labels[t] for t in chosen) if chosen else '1')
    result['metrics'].extend([{'name': label, 'value': number(value)} for label, value in [
        ('R²', model.rsquared), ('Adjusted R²', model.rsquared_adj), ('F-statistic', model.fvalue),
        ('F p-value', model.f_pvalue), ('AIC', model.aic), ('BIC', model.bic),
        ('Residual df', model.df_resid), ('Candidate fits', len(fitted))]])
    confidence = float(options.get('confidence', .95))
    if not .8 <= confidence <= .99:
        raise DataError('Confidence level must be between 0.80 and 0.99.')
    interval = model.conf_int(alpha=1-confidence)
    result['tables'].append(table(f'Coefficients ({confidence:.0%} confidence interval)', [
        {'Term': labels[name], 'Coefficient': number(model.params[name]), 'Std. error': number(model.bse[name]),
         't': number(model.tvalues[name]), 'p-value': number(model.pvalues[name]),
         'CI lower': number(interval.loc[name, 0]), 'CI upper': number(interval.loc[name, 1])}
        for name in model.params.index]))
    residuals = np.asarray(model.resid)
    exog = model.model.exog
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        result['warnings'].append('The design matrix is rank deficient. Coefficients are not uniquely identifiable; remove redundant terms.')
    normal_sample = residuals if len(residuals) <= 5000 else np.random.default_rng(42).choice(residuals, 5000, replace=False)
    normal_note = f'Uses {len(normal_sample):,} residuals' + (' sampled with seed 42.' if len(residuals) > 5000 else '.')
    tests = []
    if np.std(residuals) < 1e-12 * max(1, np.std(data[target])):
        result['warnings'].append('Residuals are nearly zero. Normality, heteroscedasticity and influence diagnostics are not meaningful for a near-perfect fit.')
    else:
        tests.extend([
            diagnostic('Shapiro–Wilk', lambda: stats.shapiro(normal_sample), normal_note),
            diagnostic('Kolmogorov–Smirnov (Lilliefors)', lambda: lilliefors(residuals),
                       'Lilliefors calibration accounts for estimating normal mean/variance from residuals.'),
            diagnostic('D’Agostino–Pearson', lambda: stats.normaltest(residuals), 'Needs at least 8 residuals.'),
            diagnostic('Breusch–Pagan', lambda: het_breuschpagan(residuals, exog)[:2], 'Koenker robust LM version.'),
        ])
        anderson = stats.anderson(normal_sample, dist='norm')
        tests.append({'Test': 'Anderson–Darling', 'Statistic': number(anderson.statistic), 'p-value': None,
                      'Notes': normal_note + ' Compare the statistic to the critical values below.'})
        result['tables'].append(table('Anderson–Darling normal critical values', [
            {'Significance %': float(level), 'Critical value': float(value), 'Reject at level': bool(anderson.statistic > value)}
            for level, value in zip(anderson.significance_level, anderson.critical_values)]))
        white_columns = exog.shape[1] * (exog.shape[1] + 1) // 2
        if white_columns * len(data) <= 10_000_000 and len(data) > white_columns:
            tests.append(diagnostic('White', lambda: het_white(residuals, exog)[:2], 'Includes squares and cross-products.'))
        else:
            tests.append({'Test': 'White', 'Statistic': None, 'p-value': None,
                          'Notes': 'Unavailable: auxiliary design needs more rows or exceeds 10 million cells. Reduce terms.'})
    tests.append({'Test': 'Durbin–Watson', 'Statistic': number(durbin_watson(residuals)), 'p-value': None,
                  'Notes': 'Descriptive statistic; near 2 suggests little first-order autocorrelation.'})
    lag = int(options.get('lag', 10))
    if not 1 <= lag <= 100:
        raise DataError('Ljung–Box lag must be between 1 and 100.')
    actual_lag = min(lag, max(1, len(data)//5))
    lb = acorr_ljungbox(residuals, lags=[actual_lag], model_df=0, return_df=True)
    tests.append({'Test': f'Ljung–Box (lag {actual_lag})', 'Statistic': number(lb.iloc[0]['lb_stat']),
                  'p-value': number(lb.iloc[0]['lb_pvalue']), 'Notes': 'OLS residual check; model_df=0, no ARMA order fitted.'})
    result['tables'].append(table('Residual and specification tests', tests))
    vif_threshold = float(options.get('vif_threshold', 5))
    if not 1 <= vif_threshold <= 100:
        raise DataError('VIF flag threshold must be between 1 and 100.')
    vifs = []
    for i, name in enumerate(model.model.exog_names):
        if name == 'Intercept':
            continue
        vif = variance_inflation_factor(exog, i)
        vifs.append({'Term': labels[name], 'VIF': number(vif), 'Flagged': bool(not np.isfinite(vif) or vif > vif_threshold),
                     'Notes': 'Infinite / exact collinearity' if not np.isfinite(vif) else ''})
    result['tables'].append(table(f'Variance inflation factors (flag > {vif_threshold:g})', vifs, ['Term', 'VIF', 'Flagged', 'Notes']))
    result['notes'].append('OLS assumes the specified mean structure. Autocorrelation checks use the current dataset row order after missing-row exclusions, not table display sorting. Arrange observations meaningfully before interpreting them.')
    result['notes'].append('Residual-test p-values are diagnostic approximations, not proof of model validity. Blank values are undefined or unavailable; consult the row notes.')
    regression_plots(model, data, result)
    return result


def regression_plots(model, data, result):
    residuals = np.asarray(model.resid)
    fitted = np.asarray(model.fittedvalues)
    influence = model.get_influence()
    standardized = np.asarray(influence.resid_studentized_internal)
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]
    indices = np.arange(len(data))
    if len(data) > PLOT_ROWS:
        # Preserve the most influential points, even in a bounded plot.
        top = np.argsort(np.nan_to_num(cooks, nan=0))[-200:]
        uniform = np.linspace(0, len(data)-1, PLOT_ROWS-200, dtype=int)
        indices = np.unique(np.concatenate([uniform, top]))
        result['notes'].append(f'Diagnostic plots show up to {PLOT_ROWS:,} rows, including the 200 largest Cook’s distances. Model fitting and influence statistics use all complete rows.')
    row_ids = (data.index.to_numpy()[indices] + 1).tolist()

    def scatter(title, x, y, x_title, y_title):
        fig = go.Figure(go.Scatter(x=np.asarray(x)[indices].tolist(), y=np.asarray(y)[indices].tolist(),
                                   customdata=row_ids, mode='markers', marker={'size': 5, 'opacity': .65},
                                   hovertemplate='Row %{customdata}<br>%{x}<br>%{y}<extra></extra>'))
        fig.update_xaxes(title=x_title); fig.update_yaxes(title=y_title)
        result['figures'].append(figure(title, fig))
    scatter('Residuals vs fitted', fitted, residuals, 'Fitted values', 'Residuals')
    finite = standardized[np.isfinite(standardized)]
    if len(finite):
        probabilities = np.linspace(.5/len(finite), 1-.5/len(finite), min(len(finite), PLOT_ROWS))
        theoretical = stats.norm.ppf(probabilities)
        observed = np.quantile(finite, probabilities)
        fig = go.Figure(go.Scatter(x=theoretical.tolist(), y=observed.tolist(), mode='markers', name='Residual quantiles', marker={'size': 4}))
        bounds = [float(theoretical.min()), float(theoretical.max())]
        fig.add_trace(go.Scatter(x=bounds, y=bounds, mode='lines', name='Normal reference', line={'dash': 'dash'}))
        fig.update_xaxes(title='Theoretical normal quantiles'); fig.update_yaxes(title='Standardized residual quantiles')
        result['figures'].append(figure('Normal Q–Q plot', fig))
    scatter('Scale–location', fitted, np.sqrt(np.abs(standardized)), 'Fitted values', '√|standardized residuals|')
    fig = go.Figure(go.Bar(x=row_ids, y=cooks[indices].tolist()))
    fig.add_hline(y=4/len(data), line_dash='dash', annotation_text='4/n heuristic')
    fig.update_xaxes(title='Original row number'); fig.update_yaxes(title='Cook’s distance')
    result['figures'].append(figure('Cook’s distance', fig))
    scatter('Residuals vs leverage', leverage, standardized, 'Leverage', 'Standardized residuals')
    biggest = np.argsort(np.nan_to_num(cooks, nan=0))[-min(20, len(data)):][::-1]
    result['tables'].append(table('Most influential observations (top 20)', [
        {'Row': int(data.index[i])+1, 'Cook’s distance': number(cooks[i]), 'Leverage': number(leverage[i]),
         'Standardized residual': number(standardized[i]), 'Cook > 4/n': bool(cooks[i] > 4/len(data))} for i in biggest]))
