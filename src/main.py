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
from feature_extractor import PSFeatureExtractor as FeatureExtractor
from config.baseline_h2c import H2CBaselineConfig, FeatureExtractorConf
from hydra.core.config_store import ConfigStore

logging.basicConfig(filename='main.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

cs = ConfigStore.instance()
cs.store(name="config", node=H2CBaselineConfig)
cs.store(group="",name="baseline_h2c", node=H2CBaselineConfig)
cs.store(group="features",name='features',node=FeatureExtractorConf)

def plot_learning_curves(estimator, X, y, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),name_conf=""):
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y.ravel(), cv=cv, n_jobs=n_jobs,train_sizes=train_sizes,return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.grid()
    plt.title('Learning Curve')
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.legend(loc="best")
    plt.savefig('output_learningcurve_'+name_conf+'.png', bbox_inches='tight')
    plt.close()

    # Plot n_samples vs fit_times
    plt.grid()
    plt.title("Scalability of the model")
    plt.xlabel("Training examples")
    plt.ylabel("fit_times")
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    plt.legend(loc="best")
    plt.savefig('output_samples_'+name_conf+'.png', bbox_inches='tight')
    plt.close()

    # Plot fit_time vs score
    plt.grid()
    plt.xlabel("fit_times")
    plt.ylabel("Score")
    plt.title("Performance of the model")
    plt.plot(fit_times_mean, test_scores_mean, 'o-',label="Testing score")
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    plt.plot(fit_times_mean, train_scores_mean, 'o-', label="Training score")
    plt.fill_between(fit_times_mean, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.legend(loc="best")
    plt.savefig('output_score_'+name_conf+'.png', bbox_inches='tight')
    plt.close()

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
    X_train,y_train, X_test, y_test = get_train_test(cfg, X, Y)

    if cfg.over_sample:
        X_train, y_train = over_sample(cfg,X_train,y_train)
        features_name += 'os'
    else:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curves(model, X, Y,
                            cv=cv, n_jobs=4,name_conf=model_name+'_'+features_name+'_'+cfg.oversampling)

    model.fit(X_train, y_train)
    plot_confusion_matrix(estimator=model, X=X_test, y_true=y_test, cmap='OrRd')
    plt.savefig('output'+model_name+'_'+features_name+'_'+cfg.oversampling+'.png',bbox_inches='tight')

    y_pred = model.predict(X_test)
    labels_name = np.unique(Y)
    logging.info('Model:{}, Out_path:{}'.format(model_name,'output'+features_name+'_'+cfg.oversampling+'.png'))
    logging.info('{}'.format(metrics.classification_report(y_test, y_pred, labels_name)))
    logging.info('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))
    logging.info('Weighted f1 score : {}'.format(f1_score(y_test, y_pred, average='weighted')))


@hydra.main(config_path="config", config_name="config")
def main(cfg: H2CBaselineConfig):
    psf_extractor = FeatureExtractor(cfg=cfg.features)
    tokens_claims, tokens_headlines = psf_extractor.tokenize()
    features, features_name = psf_extractor.generate_Features()
    logging.info('Feature set {}:{}'.format(features.shape,features_name))
    labels = np.reshape(psf_extractor.labels, (len(psf_extractor.labels), 1))
    for l in range(labels.shape[0]):
        if labels[l][0] == 'Disagree':
            labels[l][0] = "Notagree"
    # print(metrics.SCORERS.keys())
    logging.info('SVC Model')
    model = SVC(C=cfg.svc.C, class_weight='balanced', coef0=cfg.svc.coef0,
                decision_function_shape='ovo', degree=cfg.svc.degree, gamma='scale', kernel=cfg.svc.kernel,
                max_iter=-1, shrinking=True, tol=cfg.svc.tol)
    common_train_test(cfg=cfg, model=model, X=features, Y=labels,
                      features_name=features_name + 'c={}'.format(cfg.svc.C))

    logging.info('RandomForestClassifier')
    max_features=cfg.random_forest.max_features if cfg.random_forest.max_features != 'None' else None
    model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced', n_jobs=-1,
                                   criterion=cfg.random_forest.criterion,
                                   max_features=max_features,
                                   n_estimators=cfg.random_forest.n_estimators
                                   )
    common_train_test(cfg=cfg, model=model, X=features, Y=labels,
                      features_name=features_name + 'cri{}_maxfea{}_estim{}'.format(cfg.random_forest.criterion, cfg.random_forest.max_features,
                                                                                    cfg.random_forest.n_estimators))

    logging.info('LinearSVC')
    model = LinearSVC(C=cfg.linear_svc.C, class_weight='balanced', dual=True, fit_intercept=True,
                      intercept_scaling=1, loss=cfg.linear_svc.loss, max_iter=1000, multi_class='ovr',
                      penalty=cfg.linear_svc.penalty, tol=cfg.linear_svc.tol)
    common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name)

    logging.info('LogisticRegression')
    model = LogisticRegression(C=cfg.logistic_regression.C,
                               intercept_scaling=1,
                               penalty='l2',
                               solver=cfg.logistic_regression.solver, tol=0.0001)
    common_train_test(cfg=cfg, model=model, X=features, Y=labels,
                      features_name=features_name + 'l2_solv{}_c{}'.format(cfg.logistic_regression.solver, cfg.logistic_regression.C))

    logging.info('GaussianNB')
    model = GaussianNB()
    common_train_test(cfg=cfg, model=model, X=features, Y=labels, features_name=features_name)




main()
