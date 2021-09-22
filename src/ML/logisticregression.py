import pandas as pd
import numpy as np
import logging
import os.path
import joblib
import hydra
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, learning_curve, ShuffleSplit, train_test_split, cross_val_score, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, ADASYN, BorderlineSMOTE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from src.feature_extractor import PSFeatureExtractor as FeatureExtractor
from src.config.baseline_h2c import H2CBaselineConfig, FeatureExtractorConf
from hydra.core.config_store import ConfigStore
import numpy as np
logging.basicConfig(filename='main.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

cs = ConfigStore.instance()
cs.store(name="config", node=H2CBaselineConfig)
cs.store(group="",name="baseline_h2c", node=H2CBaselineConfig)
cs.store(group="features",name='features',node=FeatureExtractorConf)

def get_train_test(cfg:H2CBaselineConfig,X, Y, features_name=''):
    load_X_train = cfg.load_path + '/X_train_' + features_name + '.pkl'
    load_X_test = cfg.load_path + '/X_test_' + features_name + '.pkl'
    load_y_train = cfg.load_path + '/y_train_' + features_name + '.pkl'
    load_y_test = cfg.load_path + '/y_test_' + features_name + '.pkl'
    print(cfg.load_if_exist)
    print(load_X_train)
    if cfg.load_if_exist and os.path.isfile(load_X_train) == True:
        X_train = joblib.load(load_X_train)
        X_test = joblib.load(load_X_test)
        y_train = joblib.load(load_y_train)
        y_test = joblib.load(load_y_test)
        logging.info('X_train loaded successfully.')
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=cfg.test_size, random_state=0)
        if cfg.save_datasets:
            joblib.dump(X_train, cfg.load_path + '/X_train_' + features_name + '.pkl')
            joblib.dump(X_test, cfg.load_path + '/X_test_' + features_name + '.pkl')
            joblib.dump(y_train, cfg.load_path + '/y_train_' + features_name + '.pkl')
            joblib.dump(y_test, cfg.load_path + '/y_test_' + features_name + '.pkl')
            logging.info('Train and test sets saved successfully. Path:' + cfg.load_path + '/***_' + features_name)
    return X_train,y_train, X_test, y_test

def common_train_test(cfg:H2CBaselineConfig, model, X, Y, features_name=''):
    model_name = model.__class__.__name__
    X_train,y_train, X_test, y_test = get_train_test(cfg, X, Y)


    model.fit(X_train, y_train)
    plot_confusion_matrix(estimator=model, X=X_test, y_true=y_test, cmap='OrRd')
    plt.savefig('output'+model_name+'_'+features_name+'_'+cfg.oversampling+'.png',bbox_inches='tight')

    y_pred = model.predict(X_test)
    labels_name = np.unique(Y)
    logging.info('Model:{}, Out_path:{}'.format(model_name,'output'+features_name+'_'+cfg.oversampling+'.png'))
    logging.info('{}'.format(metrics.classification_report(y_test, y_pred, labels_name)))
    logging.info('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))
    logging.info('Weighted f1 score : {}'.format(f1_score(y_test, y_pred, average='weighted')))


@hydra.main(config_path="../config", config_name="config")
def main(cfg: H2CBaselineConfig):
    psf_extractor = FeatureExtractor(cfg=cfg.features)
    tokens_claims, tokens_headlines = psf_extractor.tokenize()
    features, features_name = psf_extractor.generate_Features()
    logging.info('Feature set {}:{}'.format(features.shape,features_name))
    labels = np.reshape(psf_extractor.labels, (len(psf_extractor.labels), 1))
    for l in range(labels.shape[0]):
        if labels[l][0] == 'Disagree':
            labels[l][0] = "Notagree"

    ranges = np.arange(0.5, 5.0, 0.5)
    erange = np.arange(0,1, 0.1)
    for c in ranges:
        for e in erange:
            logging.info('LogisticRegression---penalty=elasticnet----solver=saga c={}---erange{}'.format(c,e))
            model = LogisticRegression(C=c,
                                   intercept_scaling=1,
                                   penalty='elasticnet',
                                   solver='saga', tol=0.0001,l1_ratio=e)
            common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name+'elsticnet_c{}_e{}'.format(c,e))

    solvers = ['newton-cg', 'sag', 'lbfgs']
    for solver in solvers:
        for c in ranges:
            logging.info('LogisticRegression---penalty=l2----solver={} c={}'.format(solver, c))
            model = LogisticRegression(C=c,
                                       intercept_scaling=1,
                                       penalty='l2',
                                       solver=solver, tol=0.0001)
            common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name+'l2_solv{}_c{}'.format(solver,c))


main()
