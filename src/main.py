import pandas as pd
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import stanfordnlp
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

import os.path
import joblib

from feature_extractor import PSFeatureExtractor as FeatureExtractor

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC)
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import hydra
from config.baseline_h2c import H2CBaselineConfig, FeatureExtractorConf
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=H2CBaselineConfig)
cs.store(group="",name="baseline_h2c", node=H2CBaselineConfig)
cs.store(group="features",name='features',node=FeatureExtractorConf)
def common_train_test(cfg:H2CBaselineConfig,model, X, Y, test_size=0.2
                      , save_datasets=True, save_path='', load_if_exist=True, load_path='', additional_description=''
                      , features_name=''):
    model_name = model.__class__.__name__
    file_not_exist = False
    if load_if_exist:
        assert len(load_path) > 0, "Please enter load_path."
        load_X_train = load_path + '/X_train_' + features_name + '.pkl'
        load_X_test = load_path + '/X_test_' + features_name + '.pkl'
        load_y_train = load_path + '/y_train_' + features_name + '.pkl'
        load_y_test = load_path + '/y_test_' + features_name + '.pkl'
        if os.path.isfile(load_X_train) == True:
            X_train = joblib.load(load_X_train)
            print('X_train loaded successfully.')
        else:
            print('X_train file is not exist.')
            file_not_exist = True
        if os.path.isfile(load_X_test) == True:
            X_test = joblib.load(load_X_test)
            print('X_test loaded successfully.')
        else:
            print('X_test file is not exist.')
            file_not_exist = True
        if os.path.isfile(load_y_train) == True:
            y_train = joblib.load(load_y_train)
            print('y_train loaded successfully.')
        else:
            print('y_train file is not exist.')
            file_not_exist = True
        if os.path.isfile(load_y_test) == True:
            y_test = joblib.load(load_y_test)
            print('y_test loaded successfully.')
        else:
            print('y_test file is not exist.')
            file_not_exist = True
    if load_if_exist == False or file_not_exist == True:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=test_size, random_state=0)
        print('Train and test sets created successfully.')
        if save_datasets:
            joblib.dump(X_train, load_path + '/X_train_' + features_name + '.pkl')
            joblib.dump(X_test, load_path + '/X_test_' + features_name + '.pkl')
            joblib.dump(y_train, load_path + '/y_train_' + features_name + '.pkl')
            joblib.dump(y_test, load_path + '/y_test_' + features_name + '.pkl')
            print('Train and test sets saved successfully.')
    else:
        print('Train and test sets loaded successfully.')

    if cfg.oversampling == 'BorderlineSMOTE':
        sampling_strategy = "auto"
        ada = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=0)
        X_train, y_train = ada.fit_resample(X_train, y_train)

    elif cfg.oversampling == 'SVMSMOTE':
        sm = SVMSMOTE(random_state=0)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    elif cfg.oversampling == 'RandomOverSampler':
        rndsampler = RandomOverSampler(random_state=0)
        X_train, y_train = rndsampler.fit_resample(X_train, y_train)
    elif cfg.oversampling == 'SMOTE':
        sm = SMOTE(random_state=0)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    elif cfg.oversampling == 'ADASYN':
        adsn = ADASYN(sampling_strategy="minority"
                      , n_neighbors=cfg.N_neighbors
                      , random_state=cfg.Random_state
                      )
        X_train, y_train = adsn.fit_resample(X_train, y_train)

    mydic = {}
    for ylabel in y_train.flatten():
        if ylabel in mydic:
            mydic[ylabel] += 1
        else:
            mydic[ylabel] = 1
    print('After over smapling', mydic)
    print('Number of sampels', y_train.shape)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    labels_name = np.unique(Y)
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels_name, yticklabels=labels_name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    print(model_name)
    plt.savefig('output.png',bbox_inches='tight')

    print(metrics.classification_report(y_test, y_pred, labels_name))
    print('accuracy : ', accuracy_score(y_test, y_pred))
    print('weighted f1 score : ', f1_score(y_test, y_pred, average='weighted'))
    if len(additional_description) > 0:
        print(additional_description)
    return y_pred

@hydra.main(config_path="config", config_name="config")
def main(cfg: H2CBaselineConfig):
    psf_extractor = FeatureExtractor(cfg=cfg.features)
    tokens_claims, tokens_headlines = psf_extractor.hazm_tokenize()
    features_tfidf, features_name_tfidf = psf_extractor.generate_Features()
    print('shape features',features_tfidf.shape)
    labels = np.reshape(psf_extractor.labels, (len(psf_extractor.labels), 1))
    for l in range(labels.shape[0]):
        if labels[l][0] == 'Disagree':
            labels[l][0] = "Notagree"
    print(labels.shape)
    mydic = {}
    for l in labels:
        ll = l[0]
        if ll in mydic:
            mydic[ll] += 1
        else:
            mydic[ll] = 1
    print(mydic)
    print(metrics.SCORERS.keys())
    print("done")
    model = SVC(C=10, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0,
                decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    result = common_train_test(cfg=cfg, model=model, X=features_tfidf, Y=labels, test_size=0.2
                               , additional_description='With the features : ' + features_name_tfidf
                               , save_datasets=False, save_path=cfg.save_path
                               , load_if_exist=False, load_path=cfg.save_path
                               , features_name=features_name_tfidf)




main()
