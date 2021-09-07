import pandas as pd
import numpy as np
import logging
import os.path
import joblib
import hydra
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, learning_curve, ShuffleSplit, train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    print(train_scores_mean - train_scores_std)
    print(train_scores_mean + train_scores_std)
    print(train_scores_mean)


    axes[3].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[3].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[3].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def common_train_test(cfg:H2CBaselineConfig,model, X, Y, test_size=0.2
                      , save_datasets=True, save_path='', load_if_exist=True, load_path='', additional_description=''
                      , features_name=''):
    model_name = model.__class__.__name__
    assert len(load_path) > 0, "Please enter load_path."
    load_X = load_path + '/X_' + features_name + '.pkl'
    load_y = load_path + '/y_' + features_name + '.pkl'
    if load_if_exist and os.path.isfile(load_X) and os.path.isfile(load_y):
        X_train = joblib.load(load_X)
        y_train = joblib.load(load_y)
        print('X,Y loaded successfully.')
    else:
        joblib.dump(X, load_path + '/X_' + features_name + '.pkl')
        joblib.dump(Y, load_path + '/y_' + features_name + '.pkl')
        print('X,Y saved successfully.')

    X,Y = over_sample(cfg)

    fig, axes = plt.subplots(4, 1, figsize=(10, 15))

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    plot_learning_curve(estimator, 'ME', X_train, y_train, axes=axes, ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)
    plt.savefig('output2.png',bbox_inches='tight')

    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    #
    # conf_mat = confusion_matrix(y_test, y_pred)
    # fig, ax = plt.subplots(figsize=(4, 4))
    # labels_name = np.unique(Y)
    # sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels_name, yticklabels=labels_name)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # print(model_name)
    # plt.savefig('output.png',bbox_inches='tight')
    #
    # print(metrics.classification_report(y_test, y_pred, labels_name))
    # print('accuracy : ', accuracy_score(y_test, y_pred))
    # print('weighted f1 score : ', f1_score(y_test, y_pred, average='weighted'))
    # if len(additional_description) > 0:
    #     print(additional_description)
    # return y_pred
    return  None

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
