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

from src.feature_extractor import PSFeatureExtractor as FeatureExtractor

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC)
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import hydra
from config.baseline_h2c import H2CBaselineConfig
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=H2CBaselineConfig)
cs.store(name="baseline_h2c", node=H2CBaselineConfig)

@hydra.main(config_path="config", config_name="config")
def main(cfg: H2CBaselineConfig):
    psf_extractor = FeatureExtractor(cfg=cfg)
    print("done")
main()

#stanford_models_path  = '/content/drive/My Drive/persian_stance_baseline_data'
# dataset_path = '/content/drive/My Drive/multilingual_bert_dataset/dataset/HeadlineToClaim_multilingualBERT.csv'
# polarity_dataset_path = '/content/drive/My Drive/persian_stance_baseline_data/dataset/PerSent.xlsx'
# save_load_path = "/content/drive/My Drive/persian_stance_baseline_data/vectors"
# w2v_model_path = "/content/drive/My Drive/persian_stance_baseline_data/vectors/w2v_persian.pkl"
# train_test_sets_save_path = "/content/drive/My Drive/persian_stance_baseline_data/dataset"
# train_test_sets_load_path = "/content/drive/My Drive/persian_stance_baseline_data/dataset"
oversampling = 'ADASYN'
N_neighbors = 9
Random_state = 88


tokens_claims, tokens_headlines = psf_extractor.nltk_tokenize()


def k_fold_train_test(X, Y, k_fold, model, scoring='accuracy', additional_description=''):
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X, Y, scoring=scoring, cv=k_fold)
    result = []
    df_result = pd.DataFrame(index=range(k_fold))
    for fold_index, accuracy in enumerate(accuracies):
        result.append((model_name, fold_index, accuracy))

    df_result = pd.DataFrame(result, columns=['model_name', 'fold_index', scoring])

    sns.boxplot(x='model_name', y=scoring, data=df_result)
    sns.stripplot(x='model_name', y=scoring, data=df_result,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()

    print('Mean ' + scoring + ' of ' + model_name + ' in ' + str(k_fold) + ' fold is: ', np.average(accuracies, axis=0))
    if len(additional_description) > 0:
        print(additional_description)
    return df_result


def common_train_test(model, X, Y, test_size=0.2
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

    if oversampling == 'BorderlineSMOTE':
        sampling_strategy = "auto"
        ada = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=0)
        X_train, y_train = ada.fit_resample(X_train, y_train)

    elif oversampling == 'SVMSMOTE':
        sm = SVMSMOTE(random_state=0)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    elif oversampling == 'RandomOverSampler':
        rndsampler = RandomOverSampler(random_state=0)
        X_train, y_train = rndsampler.fit_resample(X_train, y_train)
    elif oversampling == 'SMOTE':
        sm = SMOTE(random_state=0)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    elif oversampling == 'ADASYN':
        adsn = ADASYN(sampling_strategy="minority"
                      , n_neighbors=N_neighbors
                      , random_state=Random_state, ratio="minority"
                      )
        X_train, y_train = adsn.fit_resample(X_train, y_train)

    mydic = {}
    print(y_train)
    for l in y_train:
        ll = l[0]
        if ll in mydic:
            mydic[ll] += 1
        else:
            mydic[ll] = 1
    print(mydic)
    print(y_train.shape)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    labels_name = np.unique(Y)
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels_name, yticklabels=labels_name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    print(model_name)
    plt.show()

    print(metrics.classification_report(y_test, y_pred, labels_name))
    print('accuracy : ', accuracy_score(y_test, y_pred))
    print('weighted f1 score : ', f1_score(y_test, y_pred, average='weighted'))
    if len(additional_description) > 0:
        print(additional_description)
    return y_pred


# features_tfidf, features_name_tfidf = psf_extractor.generate_Features(w2v_model_path = w2v_model_path,save_path = save_load_path
#                                                           , save_feature= False
#                                                           , load_path= save_load_path
#                                                           , root_distance = False
#                                                           , load_if_exist = False, bow = False, w2v = False, polarity= False)
# #
# labels = np.reshape(psf_extractor.labels,(len(psf_extractor.labels),1))
# print(labels.shape)
# for l in range(labels.shape[0]):
#     if labels[l][0] == 'Disagree':
#         labels[l][0] = "Notagree"
# mydic={}
# for l in labels:
#     ll = l[0]
#     if ll in mydic:
#         mydic[ll]+=1
#     else:
#         mydic[ll]=1
# print(mydic)
#
#
# a = ["Discuss"]
# a = np.array(a*labels.shape[0])
# print(f1_score(labels,a,labels=['Discuss', 'Notagree', 'Agree', 'Unrelated'],average='weighted'))
# print(accuracy_score(labels,a))