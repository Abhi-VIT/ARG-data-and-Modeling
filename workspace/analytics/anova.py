import html
import numpy as np
import pandas as pd
import pingouin as pg
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from .common import report, selected, complete_rows, dataframe_table, table, figure, sample_plot
from ..ingestion import DataError


def anova(frame, options, progress):
    result = report('Analysis of variance', 'anova')
    kind = options.get('method', 'oneway')
    if kind not in {'oneway', 'twoway', 'welch', 'repeated'}:
        raise DataError('Choose one-way, two-way, Welch or repeated-measures ANOVA.')
    selected_data = selected(frame, options, result, minimum=2)
    target, group = options.get('target'), options.get('group')
    second, subject = options.get('group2'), options.get('subject')
    roles = [target, group]
    if kind == 'twoway':
        roles.append(second)
    if kind == 'repeated':
        roles.append(subject)
    if any(c not in selected_data for c in roles) or len(set(roles)) != len(roles):
        raise DataError('Assign distinct selected columns to the outcome, factor(s), and subject roles.')
    if not pd.api.types.is_numeric_dtype(selected_data[target]):
        raise DataError('The ANOVA outcome must be numeric.')
    if options.get('tukey') and kind in {'welch', 'repeated'}:
        raise DataError('Tukey HSD assumes independent groups and equal variance. Use it with ordinary one/two-way ANOVA.')
    alpha = float(options.get('alpha', .05))
    if not .001 <= alpha <= .2:
        raise DataError('Choose alpha between 0.001 and 0.2.')
    unused = [c for c in selected_data if c not in roles]
    if unused:
        result['notes'].append('Selected columns without an assigned role were not used: ' + ', '.join(unused))
    result['selection']['used'] = roles
    data = complete_rows(selected_data[roles], result)
    if len(data) < 4 or data[target].nunique() < 2:
        raise DataError('ANOVA needs at least 4 complete rows and a nonconstant outcome.')
    for factor in [group] + ([second] if kind == 'twoway' else []):
        if not 2 <= data[factor].nunique() <= 30:
            raise DataError('Each factor must have 2–30 observed levels.')
    internal = pd.DataFrame({'response': data[target].astype(float), 'factor1': data[group].astype(str)}, index=data.index)
    if kind == 'twoway':
        internal['factor2'] = data[second].astype(str)
        counts = pd.crosstab(internal['factor1'], internal['factor2'])
        if (counts < 2).any().any():
            raise DataError('Two-way ANOVA requires at least two observations in every factor combination.')
        if counts.size > 100 or counts.size * len(internal) > 10_000_000:
            raise DataError('Two-way design is too large. Use fewer factor levels or observations.')
    elif kind != 'repeated' and internal.groupby('factor1').size().min() < 2:
        raise DataError('Each independent group needs at least two observations.')
    progress(45, 'Fitting ANOVA and checking the design')
    if kind == 'welch':
        variances = internal.groupby('factor1')['response'].var()
        if (variances <= 0).any():
            raise DataError('Welch ANOVA requires nonzero variance within every group.')
        output = pg.welch_anova(data=internal, dv='response', between='factor1')
        output['Source'] = group
        result['tables'].append(dataframe_table('Welch ANOVA', output))
        result['notes'].append('Welch ANOVA allows unequal group variances. Tukey HSD is not applicable to this variance model.')
    elif kind == 'repeated':
        internal['subject'] = data[subject].astype(str)
        if internal.duplicated(['subject', 'factor1']).any():
            raise DataError('Repeated measures needs exactly one observation per subject × condition. Aggregate duplicates explicitly first.')
        wide = internal.pivot(index='subject', columns='factor1', values='response')
        balanced = wide.dropna()
        excluded_subjects = selected_data[subject].dropna().astype(str).nunique() - len(balanced)
        if len(balanced) < 3:
            raise DataError('Repeated measures needs at least 3 subjects with every condition observed.')
        internal = internal[internal.subject.isin(balanced.index)].copy()
        result['metrics'].extend([{'name': 'Complete subjects', 'value': len(balanced)},
                                  {'name': 'Subjects excluded', 'value': excluded_subjects}])
        # Replace row count with the actual balanced data used by rm_anova.
        for metric in result['metrics']:
            if metric['name'] == 'Rows used':
                metric['value'] = len(internal)
            elif metric['name'] == 'Rows excluded':
                metric['value'] = len(selected_data) - len(internal)
        if excluded_subjects:
            result['warnings'].append(f'{excluded_subjects} subjects with incomplete conditions were excluded listwise.')
        output = pg.rm_anova(data=internal, dv='response', within='factor1', subject='subject',
                             correction=True, detailed=True)
        output['Source'] = output['Source'].replace({'factor1': group})
        result['tables'].append(dataframe_table('Repeated-measures ANOVA', output))
        spher, w, chi2, dof, pval = pg.sphericity(balanced)
        result['tables'].append(table('Sphericity (Mauchly)', [{'Sphericity met': bool(spher),
            'W': w, 'Chi-square': chi2, 'df': dof, 'p-value': pval}]))
        result['notes'].append('One within-subject factor in long format. p-unc is uncorrected; p-GG-corr applies Greenhouse–Geisser correction when available. Sphericity is automatic for two levels. Tukey HSD is not valid for these paired observations.')
    else:
        formula = 'response ~ C(factor1)' + (' * C(factor2)' if kind == 'twoway' else '')
        model = smf.ols(formula, internal, eval_env=-1).fit()
        output = anova_lm(model, typ=2).reset_index().rename(columns={'index': 'Source', 'PR(>F)': 'p-value'})
        names = {'C(factor1)': group, 'C(factor2)': second, 'C(factor1):C(factor2)': f'{group} × {second}'}
        output['Source'] = output['Source'].replace(names)
        result['tables'].append(dataframe_table('ANOVA (type II sums of squares)', output))
        result['notes'].append('Ordinary ANOVA assumes independent observations, normally distributed errors, and equal within-group variances. Two-way models include the interaction; interpret main effects cautiously when interaction is present.')
        if options.get('tukey'):
            factor = options.get('tukey_factor', group)
            if factor not in ([group, second] if kind == 'twoway' else [group]):
                raise DataError('Choose an assigned factor for Tukey HSD.')
            key = 'factor1' if factor == group else 'factor2'
            comparison = pairwise_tukeyhsd(internal['response'], internal[key], alpha=alpha)
            i, j = np.triu_indices(len(comparison.groupsunique), 1)
            result['tables'].append(table(f'Tukey HSD · {factor}', [
                {'Group A': str(comparison.groupsunique[a]), 'Group B': str(comparison.groupsunique[b]),
                 'Mean difference (B−A)': float(comparison.meandiffs[k]), 'Adjusted p-value': float(comparison.pvalues[k]),
                 'CI lower': float(comparison.confint[k, 0]), 'CI upper': float(comparison.confint[k, 1]),
                 'Reject': bool(comparison.reject[k])} for k, (a, b) in enumerate(zip(i, j))]))
            if kind == 'twoway':
                result['warnings'].append('Tukey compares marginal groups of the chosen factor, pooling the other factor. It does not estimate simple effects within an interaction.')
    grouping = ['factor1', 'factor2'] if kind == 'twoway' else ['factor1']
    summary = internal.groupby(grouping, observed=True)['response'].agg(['count', 'mean', 'std']).reset_index()
    summary = summary.rename(columns={'factor1': f'Factor 1 ({group})', 'factor2': f'Factor 2 ({second})'})
    result['tables'].append(dataframe_table('Group summaries', summary))
    plot_data = sample_plot(internal, result)
    fig = go.Figure()
    if kind == 'twoway':
        for level, subset in plot_data.groupby('factor2'):
            fig.add_trace(go.Box(x=[html.escape(v) for v in subset.factor1], y=subset.response.tolist(), name=html.escape(str(level))))
        fig.update_layout(boxmode='group')
    else:
        for level, subset in plot_data.groupby('factor1'):
            fig.add_trace(go.Box(y=subset.response.tolist(), name=html.escape(str(level))))
    fig.update_yaxes(title=html.escape(target))
    result['figures'].append(figure(f'{target} by {group}', fig))
    result['notes'].append(f'Alpha = {alpha:g}. ANOVA p-values are unadjusted; Tukey p-values, when requested, are familywise adjusted.')
    return result
