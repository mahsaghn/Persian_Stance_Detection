import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
import joblib
import hydra

from feature_extractor import PSFeatureExtractor as FeatureExtractor
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, ADASYN, BorderlineSMOTE
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, learning_curve, ShuffleSplit
from hydra.core.config_store import ConfigStore
from config.baseline_h2c import H2CBaselineConfig, FeatureExtractorConf

logging.basicConfig(filename='main.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

cs = ConfigStore.instance()
cs.store(name="config", node=H2CBaselineConfig)
cs.store(group="",name="baseline_h2c", node=H2CBaselineConfig)
cs.store(group="features",name='features',node=FeatureExtractorConf)


def over_sample(cfg,X,Y):
    mydic = {}
    for ylabel in Y.flatten():
        if ylabel in mydic:
            mydic[ylabel] += 1
        else:
            mydic[ylabel] = 1
    logging.info('Before oversmapling Y shape:{}, Class distribution:{}'.format(Y.shape,mydic))
    logging.info('Oversampling Strategy: '+ cfg.oversampling)
    if cfg.oversampling == 'BorderlineSMOTE':
        sampling_strategy = "auto"
        ada = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=0)
        X_n, Y_n = ada.fit_resample(X, Y)
    elif cfg.oversampling == 'SVMSMOTE':
        sm = SVMSMOTE(random_state=0)
        X_n, Y_n = sm.fit_resample(X, Y)
    elif cfg.oversampling == 'RandomOverSampler':
        rndsampler = RandomOverSampler(random_state=0)
        X_n, Y_n = rndsampler.fit_resample(X, Y)
    elif cfg.oversampling == 'SMOTE':
        sm = SMOTE(random_state=0)
        X_n, Y_n = sm.fit_resample(X, Y)
    elif cfg.oversampling == 'ADASYN':
        adsn = ADASYN(sampling_strategy="minority"
                      , n_neighbors=cfg.N_neighbors
                      , random_state=cfg.Random_state
                      )
        X_n, Y_n = adsn.fit_resample(X, Y)

    mydic = {}
    for ylabel in Y_n.flatten():
        if ylabel in mydic:
            mydic[ylabel] += 1
        else:
            mydic[ylabel] = 1
    logging.info('After oversmapling Y shape:{}, Class distribution:{}'.format(Y_n.shape,mydic))
    return  X_n, Y_n


def common_train_test(cfg:H2CBaselineConfig, model, X, Y, features_name=''):
    model_name = model.__class__.__name__
    load_X_train = cfg.load_path + '/X_train_' + features_name + '.pkl'
    load_X_test = cfg.load_path + '/X_test_' + features_name + '.pkl'
    load_y_train = cfg.load_path + '/y_train_' + features_name + '.pkl'
    load_y_test = cfg.load_path + '/y_test_' + features_name + '.pkl'
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
            logging.info('Train and test sets saved successfully. Path:'+ cfg.load_path+'/***_'+features_name )

    if cfg.over_sample:
        X_train, y_train = over_sample(cfg,X_train,y_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    labels_name = np.unique(Y)
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels_name, yticklabels=labels_name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('output'+model_name+'_'+features_name+'_'+cfg.oversampling+'.png',bbox_inches='tight')
    logging.info('Model:{}, Out_path:{}'.format(model_name,'output'+features_name+'_'+cfg.oversampling+'.png'))
    logging.info('{}'.format(metrics.classification_report(y_test, y_pred, labels_name)))
    logging.info('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))
    logging.info('Weighted f1 score : {}'.format(f1_score(y_test, y_pred, average='weighted')))

@hydra.main(config_path="config", config_name="config")
def main(cfg: H2CBaselineConfig):
    psf_extractor = FeatureExtractor(cfg=cfg.features)
    tokens_claims, tokens_headlines = psf_extractor.nltk_tokenize()
    features, features_name = psf_extractor.generate_Features()
    logging.info('Feature set {}:{}'.format(features.shape,features_name))
    labels = np.reshape(psf_extractor.labels, (len(psf_extractor.labels), 1))
    for l in range(labels.shape[0]):
        if labels[l][0] == 'Disagree':
            labels[l][0] = "Notagree"
    # print(metrics.SCORERS.keys())
    logging.info('SVC Model')
    model = SVC(C=10, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0,
                decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name)

    logging.info('RandomForestClassifier')
    model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                   class_weight='balanced_subsample', criterion='entropy',
                                   max_depth=None, max_features='auto', max_leaf_nodes=None,
                                   max_samples=None, min_impurity_decrease=0.0,
                                   min_impurity_split=None, min_samples_leaf=1,
                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                   n_estimators=75, n_jobs=None, oob_score=False,
                                   random_state=None, verbose=0, warm_start=False)
    common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name)

    logging.info('LinearSVC')
    model = LinearSVC(C=0.5, class_weight='balanced', dual=True, fit_intercept=True,
                      intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
                      penalty='l2', random_state=None, tol=0.0001, verbose=0)
    common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name)

    logging.info('LogisticRegression')
    model = LogisticRegression(C=1, class_weight='balanced', dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=1000,
                               multi_class='ovr', n_jobs=None, penalty='l1',
                               random_state=None, solver='saga', tol=0.0001, verbose=0,
                               warm_start=False)
    common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name)

    logging.info('GaussianNB')
    model = GaussianNB()
    common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name)




main()
logging.info('Finish')
