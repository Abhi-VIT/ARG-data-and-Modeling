import json
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy import stats
from django.test import SimpleTestCase
from workspace.analytics import analyze
from workspace.cleaning import apply_operation
from workspace.ingestion import DataError


def find_table(report, prefix):
    return next(t['rows'] for t in report['tables'] if t['title'].startswith(prefix))


class AnalyticsTests(SimpleTestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        x, z = rng.normal(size=(2, 120))
        self.frame = pd.DataFrame({'x': x, 'z': z, 'y': 2 + 3*x + rng.normal(0, .25, 120),
                                   'group': np.tile(['a', 'b', 'c'], 40)})

    def run_analysis(self, kind, frame=None, **options):
        frame = self.frame if frame is None else frame
        options.setdefault('columns', list(frame))
        result = analyze(frame, dict(type=kind, **options))
        json.dumps(result, allow_nan=False)
        return result

    def test_numeric_correlations_match_scipy_and_exclude_text(self):
        for method, fn in [('pearson', stats.pearsonr), ('spearman', stats.spearmanr), ('kendall', stats.kendalltau)]:
            with self.subTest(method=method):
                result = self.run_analysis('correlation', method=method)
                row = find_table(result, 'Pairwise methods')[0]
                expected = fn(self.frame.x, self.frame.z)
                self.assertAlmostEqual(row['Coefficient'], expected.statistic)
                self.assertAlmostEqual(row['p-value'], expected.pvalue)
                self.assertEqual(result['selection']['excluded'], ['group'])

    def test_categorical_associations_and_auto(self):
        frame = pd.DataFrame({'a': ['no','yes','yes','no']*20, 'b': ['yes','yes','no','no']*20,
                              'n': np.arange(80), 'c': ['a','b','c','a']*20})
        for method, cols in [('phi',['a','b']), ('point_biserial',['a','n']), ('cramers_v',['a','c'])]:
            result = self.run_analysis('correlation', frame, method=method, columns=cols)
            row = find_table(result, 'Pairwise methods')[0]
            if method == 'point_biserial':
                expected = stats.pointbiserialr((frame.a == 'yes').astype(int), frame.n)
                self.assertAlmostEqual(row['Coefficient'], expected.statistic)
                self.assertAlmostEqual(row['p-value'], expected.pvalue)
            else:
                chi, p, _, _ = stats.chi2_contingency(pd.crosstab(frame[cols[0]],frame[cols[1]]), correction=False)
                self.assertAlmostEqual(abs(row['Coefficient']), np.sqrt(chi/80))
                self.assertAlmostEqual(row['p-value'], p)
            auto = self.run_analysis('correlation', frame, method='auto', columns=cols)
            self.assertEqual(find_table(auto, 'Pairwise methods')[0]['Used'], method)

    def test_pairwise_missing_constant_and_explicit_selection(self):
        frame = pd.DataFrame({'x':[1,2,3,np.nan,5], 'y':[2,np.nan,6,8,10], 'constant':[1]*5})
        result = self.run_analysis('correlation', frame, method='pearson')
        rows = find_table(result, 'Pairwise methods')
        self.assertEqual(rows[0]['Paired n'],3)
        self.assertAlmostEqual(rows[0]['Coefficient'],1)
        self.assertIsNone(rows[1]['Coefficient'])
        for cols in [[], ['unknown']]:
            with self.assertRaises(DataError): self.run_analysis('descriptive', columns=cols)

    def test_point_biserial_override_handles_numeric_binary_first(self):
        frame=pd.DataFrame({'numeric':[0,1,1,0]*10,'binary':['a','b','a','b']*10})
        result=self.run_analysis('correlation',frame,method='point_biserial')
        row=find_table(result,'Pairwise methods')[0]
        self.assertIsNotNone(row['Coefficient'])
        self.assertAlmostEqual(row['Coefficient'],stats.pointbiserialr((frame.binary=='b').astype(int),frame.numeric).statistic)

    def test_regression_matches_ols_and_has_diagnostics(self):
        result = self.run_analysis('regression', target='y')
        expected = sm.OLS(self.frame.y, sm.add_constant(self.frame[['x','z']])).fit()
        rows = find_table(result, 'Coefficients')
        for i,row in enumerate(rows):
            self.assertAlmostEqual(row['Coefficient'], expected.params.iloc[i])
            self.assertAlmostEqual(row['CI lower'], expected.conf_int().iloc[i,0])
            self.assertAlmostEqual(row['p-value'], expected.pvalues.iloc[i])
        metrics = {m['name']:m['value'] for m in result['metrics']}
        self.assertAlmostEqual(metrics['R²'],expected.rsquared)
        tests = find_table(result, 'Residual and')
        self.assertEqual(len(tests),8)
        self.assertEqual(len(result['figures']),5)
        self.assertEqual(len(find_table(result, 'Variance inflation')),2)

    def test_custom_regression_uses_safe_aliases_and_hierarchical_terms(self):
        dangerous = "x'); __import__('os').system('echo unsafe') #"
        frame = self.frame.rename(columns={'x':dangerous})
        result = self.run_analysis('regression', frame, target='y', model='custom',
                                   polynomials={dangerous:3}, interactions=[[dangerous,'z']])
        terms = [r['Term'] for r in find_table(result,'Coefficients')]
        self.assertIn(dangerous+'^2',terms)
        self.assertIn(dangerous+'^3',terms)
        self.assertIn(dangerous+' × z',terms)
        for options in [{'polynomials': {'x':4}}, {'interactions':[['x','missing']]}]:
            with self.assertRaises(DataError): self.run_analysis('regression',target='y',model='custom',**options)

    def test_stepwise_directions_and_criteria(self):
        for direction in ['forward','backward','both']:
            for criterion in ['aic','bic','pvalue']:
                with self.subTest(direction=direction,criterion=criterion):
                    result=self.run_analysis('regression',target='y',stepwise=direction,criterion=criterion)
                    self.assertIn('x',[r['Term'] for r in find_table(result,'Coefficients')])
                    self.assertTrue(any('exploratory' in n for n in result['notes']))

    def test_collinear_perfect_and_small_regression(self):
        frame = self.frame[['x','y']].copy(); frame['duplicate']=frame.x*2
        result=self.run_analysis('regression',frame,target='y')
        self.assertTrue(any('rank deficient' in w for w in result['warnings']))
        self.assertTrue(all(r['Flagged'] for r in find_table(result,'Variance inflation')))
        frame['y']=2+3*frame.x
        result=self.run_analysis('regression',frame,target='y')
        self.assertTrue(any('nearly zero' in w for w in result['warnings']))
        with self.assertRaises(DataError): self.run_analysis('regression',frame.iloc[:3],target='y')

    def test_oneway_welch_and_tukey(self):
        for method in ['oneway','welch']:
            result=self.run_analysis('anova',target='y',group='group',method=method,tukey=method=='oneway')
            row=result['tables'][0]['rows'][0]
            if method=='oneway':
                expected=stats.f_oneway(*(g.y for _,g in self.frame.groupby('group')))
                self.assertAlmostEqual(row['F'],expected.statistic)
                self.assertAlmostEqual(row['p-value'],expected.pvalue)
                self.assertEqual(len(find_table(result,'Tukey')),3)
            else:
                expected=pg.welch_anova(data=self.frame,dv='y',between='group').iloc[0]
                self.assertAlmostEqual(row['F'],expected['F'])
        with self.assertRaises(DataError):
            self.run_analysis('anova',target='y',group='group',method='welch',tukey=True)

    def test_two_way_anova(self):
        frame=self.frame.copy();frame['factor2']=np.tile(['low']*3+['high']*3,20)
        result=self.run_analysis('anova',frame,target='y',group='group',group2='factor2',method='twoway',tukey=True)
        self.assertEqual(len(result['tables'][0]['rows']),4)
        self.assertIn('group × factor2',[r['Source'] for r in result['tables'][0]['rows']])

    def test_repeated_measures_balancing_two_and_three_levels(self):
        rng=np.random.default_rng(4)
        for levels in [2,3]:
            frame=pd.DataFrame({'subject':np.repeat(np.arange(12),levels), 'condition':list(range(levels))*12,
                                'y':rng.normal(size=12*levels)})
            expected=pg.rm_anova(data=frame,dv='y',within='condition',subject='subject',detailed=True,correction=True)
            result=self.run_analysis('anova',frame,target='y',group='condition',subject='subject',method='repeated')
            self.assertAlmostEqual(result['tables'][0]['rows'][0]['F'],expected.iloc[0]['F'])
            frame.loc[0,'y']=np.nan
            result=self.run_analysis('anova',frame,target='y',group='condition',subject='subject',method='repeated')
            self.assertEqual(next(m['value'] for m in result['metrics'] if m['name']=='Complete subjects'),11)
            duplicate=pd.concat([frame,frame.iloc[[1]]],ignore_index=True)
            with self.assertRaises(DataError):
                self.run_analysis('anova',duplicate,target='y',group='condition',subject='subject',method='repeated')

    def test_all_eda_plots_and_descriptive(self):
        for kind in ['histogram','box','violin','scatter','pair','heatmap','bar','line']:
            with self.subTest(kind=kind):
                result=self.run_analysis('eda',plot=kind,x='group' if kind=='bar' else 'x',y='y')
                self.assertTrue(result['figures'])
        result=self.run_analysis('descriptive')
        rows=find_table(result,'Column summaries')
        self.assertAlmostEqual(rows[0]['Mean'],self.frame.x.mean())
        self.assertEqual(rows[-1]['Mode'],'a')

    def test_plot_sampling_is_bounded_but_statistics_use_all_rows(self):
        frame=pd.DataFrame({'x':np.arange(6000),'group':['a']*6000})
        result=self.run_analysis('eda',frame,plot='histogram')
        self.assertEqual(len(result['figures'][0]['figure']['data'][0]['x']),5000)
        result=self.run_analysis('eda',frame,plot='bar',x='group')
        self.assertEqual(find_table(result,'Category counts')[0]['Count'],6000)
        self.assertFalse(any('sample' in n for n in result['notes']))

    def test_drop_selected_columns_and_guards(self):
        result=apply_operation(self.frame,{'action':'drop_selected_columns','selection':{'columns':['x','group']}})
        self.assertEqual(list(result),['z','y'])
        for scope in [{},{'columns':[]},{'columns':list(self.frame)},{'columns':['x'],'rows':[0]}]:
            with self.assertRaises(DataError):
                apply_operation(self.frame,{'action':'drop_selected_columns','selection':scope})
