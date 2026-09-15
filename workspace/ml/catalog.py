"""Whitelisted estimators and bounded hyperparameters, also used to render the UI."""
import math
from ..ingestion import DataError


def integer(default, low, high):
    return {'type':'int','default':default,'min':low,'max':high,'step':1}


def real(default, low, high, step=.01):
    return {'type':'float','default':default,'min':low,'max':high,'step':step}


def choice(default, values):
    return {'type':'choice','default':default,'values':values}


ALPHA = real(1, .0001, 100, .1)
TREE = {'max_depth':integer(8,1,40),'min_samples_split':integer(2,2,30),'min_samples_leaf':integer(1,1,30)}
FOREST = {**TREE,'n_estimators':integer(100,10,300),'max_features':choice('sqrt',['sqrt','log2',1.0])}
BOOST = {'n_estimators':integer(100,10,300),'learning_rate':real(.1,.01,1),'max_depth':integer(3,1,12)}
KNN = {'n_neighbors':integer(5,1,50),'weights':choice('uniform',['uniform','distance']),'p':choice(2,[1,2])}
CATALOG = {}


def register(key, name, task, params):
    CATALOG[key]={'key':key,'name':name,'task':task,'params':params}


register('linear','Linear regression','regression',{'fit_intercept':choice(True,[True,False])})
register('ridge','Ridge','regression',{'alpha':ALPHA})
register('lasso','Lasso','regression',{'alpha':ALPHA})
register('elasticnet','ElasticNet','regression',{'alpha':ALPHA,'l1_ratio':real(.5,0,1)})
register('polynomial','Polynomial regression','regression',{'degree':integer(2,2,3),'fit_intercept':choice(True,[True,False])})
register('svr','Support vector regression','regression',{'C':real(1,.01,100,.1),'epsilon':real(.1,.001,2),'kernel':choice('rbf',['rbf','linear','poly']),'gamma':choice('scale',['scale','auto'])})
for suffix, task in [('reg','regression'),('class','classification')]:
    register('tree_'+suffix,'Decision tree',task,TREE)
    register('forest_'+suffix,'Random forest',task,FOREST)
    register('gradient_'+suffix,'Gradient boosting',task,BOOST)
    register('xgb_'+suffix,'XGBoost',task,{**BOOST,'subsample':real(1,.5,1),'colsample_bytree':real(1,.5,1)})
    register('lgbm_'+suffix,'LightGBM',task,{'n_estimators':integer(100,10,300),'learning_rate':real(.1,.01,1),'num_leaves':integer(31,2,100),'min_child_samples':integer(20,2,100)})
    register('knn_'+suffix,'K-nearest neighbors',task,KNN)
register('logistic','Logistic regression','classification',{'C':real(1,.01,100,.1),'class_weight':choice(None,[None,'balanced']),'max_iter':integer(500,100,2000)})
register('svm','Support vector classifier','classification',{'C':real(1,.01,100,.1),'kernel':choice('rbf',['rbf','linear','poly']),'gamma':choice('scale',['scale','auto']),'class_weight':choice(None,[None,'balanced'])})
register('gaussian_nb','Gaussian naive Bayes','classification',{'var_smoothing':real(1e-9,1e-12,.1,1e-9)})
register('multinomial_nb','Multinomial naive Bayes','classification',{'alpha':real(1,.001,10,.1),'fit_prior':choice(True,[True,False])})
register('kmeans','K-Means','clustering',{'n_clusters':integer(3,2,15),'n_init':integer(10,1,20),'max_iter':integer(300,100,500)})
register('hierarchical','Hierarchical / agglomerative','clustering',{'n_clusters':integer(3,2,15),'linkage':choice('ward',['ward','complete','average','single'])})
register('dbscan','DBSCAN','clustering',{'eps':real(.5,.01,10),'min_samples':integer(5,2,50)})
register('pca','Principal component analysis','reduction',{'n_components':integer(2,2,20),'whiten':choice(False,[False,True])})
register('tsne','t-SNE','reduction',{'n_components':choice(2,[2,3]),'perplexity':real(30,2,100,1),'learning_rate':real(200,10,1000,10),'max_iter':integer(500,250,1500)})
register('umap','UMAP','reduction',{'n_components':choice(2,[2,3]),'n_neighbors':integer(15,2,100),'min_dist':real(.1,0,.99)})


def validate_params(key, supplied):
    if key not in CATALOG or not isinstance(supplied,dict):
        raise DataError('Choose a supported model and parameter values.')
    specs=CATALOG[key]['params']
    if not set(supplied)<=set(specs):
        raise DataError('Unsupported hyperparameter. Use the model controls.')
    values={name:spec['default'] for name,spec in specs.items()}
    for name,value in supplied.items():
        spec=specs[name]
        if spec['type']=='choice':
            if not any(type(value) is type(item) and value==item for item in spec['values']):
                raise DataError(f'Choose a listed value for {name}.')
        elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not spec['min']<=value<=spec['max'] or (spec['type']=='int' and int(value)!=value):
            raise DataError(f'{name} must be between {spec["min"]} and {spec["max"]}.')
        values[name]=int(value) if spec['type']=='int' else value
    return values


def estimator(key, params, seed):
    from sklearn import linear_model, tree, ensemble, neighbors, svm, naive_bayes, cluster, decomposition, manifold
    constructors={'linear':linear_model.LinearRegression,'ridge':linear_model.Ridge,'lasso':linear_model.Lasso,
        'elasticnet':linear_model.ElasticNet,'polynomial':linear_model.LinearRegression,
        'svr':svm.SVR,'tree_reg':tree.DecisionTreeRegressor,'tree_class':tree.DecisionTreeClassifier,
        'forest_reg':ensemble.RandomForestRegressor,'forest_class':ensemble.RandomForestClassifier,
        'gradient_reg':ensemble.GradientBoostingRegressor,'gradient_class':ensemble.GradientBoostingClassifier,
        'knn_reg':neighbors.KNeighborsRegressor,'knn_class':neighbors.KNeighborsClassifier,
        'logistic':linear_model.LogisticRegression,'svm':svm.SVC,'gaussian_nb':naive_bayes.GaussianNB,
        'multinomial_nb':naive_bayes.MultinomialNB,'kmeans':cluster.KMeans,'hierarchical':cluster.AgglomerativeClustering,
        'dbscan':cluster.DBSCAN,'pca':decomposition.PCA,'tsne':manifold.TSNE}
    options=dict(params)
    options.pop('degree',None)
    if key.startswith('xgb_'):
        from xgboost import XGBRegressor, XGBClassifier
        return (XGBRegressor if key.endswith('reg') else XGBClassifier)(**options,random_state=seed,n_jobs=1,tree_method='hist')
    if key.startswith('lgbm_'):
        from lightgbm import LGBMRegressor, LGBMClassifier
        return (LGBMRegressor if key.endswith('reg') else LGBMClassifier)(**options,random_state=seed,n_jobs=1,verbosity=-1)
    if key=='umap':
        from umap import UMAP
        return UMAP(**options,random_state=seed,transform_seed=seed,n_jobs=1,init='random',n_epochs=200)
    base=constructors[key]()
    available=base.get_params()
    if 'random_state' in available: options['random_state']=seed
    if 'n_jobs' in available: options['n_jobs']=1
    if key in {'lasso','elasticnet'}:options['max_iter']=5000
    if key=='svm':options['probability']=True
    if key=='hierarchical':options['compute_distances']=True
    if key=='pca':options['svd_solver']='full'
    return constructors[key](**options)
