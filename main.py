import pandas as pd
import argparse
import pickle
#import suite
import urllib
import random
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

random.seed(31)

def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[[self.column]]

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[self.column]
#Read command-line options
parser = argparse.ArgumentParser()
#parser.add_argument('--save_dir', type=str)
parser.add_argument('--train_set', type=str)
parser.add_argument('--test_set', type=str)
#parser.add_argument('--pretrain_path', type=str)
parser.add_argument('--train_or_test', type=str)
#parser.add_argument('--attn', type=str)
parser.add_argument('--ground_truth', type=str)
#parser.add_argument('--embedding_dim', type=int)
#parser.add_argument('--hidden_size', type=int)
#parser.add_argument('--layers', type=int)
#parser.add_argument('--batch_size', type=int)
#parser.add_argument('--learning_rate', type=float)
#parser.add_argument('--epochs', type=int)
#parser.add_argument('--eval_metric', type=str)
parser.add_argument('--model', type=str)
#parser.add_argument('--vector', type=str)

args = parser.parse_args()
train_data = args.train_set
test_data = args.test_set
model_name = args.model
tt = args.train_or_test
gold_labels = args.ground_truth

#load data
if ("context_matters" in train_data) and ("context_matters" in test_data):
    df = pd.read_csv(train_data+"test.csv")
    train_df = df.comments.values
    label = df.label.values
    X_train, X_test, y_train, y_test = train_test_split(train_df, label, test_size=0.2, random_state=31)
else:
    train_df = pd.read_csv(train_data + "train.csv", encoding='utf-8')
    test_df = pd.read_csv(test_data + "test.csv", encoding='utf-8')
    train_df = train_df.dropna()
    #df_train = train_df.comments.values
    if 'github' in train_data:
        df_train = train_df[['comments', 'polarity', 'subjectivity', 'num_mention', 'nltk_score', 'perspective_score', 'stanford_polite']]
    else:
        df_train = train_df.comments.values
    #X_test = test_df[['comments', 'polarity', 'subjectivity', 'num_mention', 'nltk_score', 'perspective_score', 'stanford_polite']]
    y_train = train_df.label.values
    y_test = test_df.label.values

    if 'wiki' in train_data:
        print(test_df.head())
        train_df['label'] = train_df['label'].map({True: 1, False: 0})
        y_train = train_df['label'].values

    if 'wiki' in test_data:
        test_df['label'] = test_df['label'].map({True: 1, False: 0})
        y_test = test_df['label'].values

    if 'github' in train_data:
        if train_df['label'].dtype != np.number:
            train_df['label'] = train_df['label'].map({'y': 1, 'n': 0})
        y_train = train_df['label'].values

    if 'github' in test_data:
        if test_df['label'].dtype != np.number:
            test_df['label'] = test_df['label'].map({'y': 1, 'n': 0})
        y_test = test_df['label'].values
    label = y_train
    X_train = df_train
    X_test = test_df.comments.values
    if ('github' in train_data) and ('github' in test_data):
        X_train, X_test, y_train, y_test = train_test_split(df_train, label, test_size=0.2, random_state=31)
    #X_train, X_test, y_train, y_test = train_test_split(df_train, label, test_size=0.2, random_state=31)
    print(len(y_test))
#load model
'''
model = pickle.load(open("models/"+model_name ,"rb"))
print(suite.get_prediction(text,model))
'''
if model_name == 'wiki':
    clf = Pipeline([
        ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', LogisticRegression()),
    ])
    #print(clf.predict_proba(text)[:, 1])
    print(X_train.shape, y_train.shape, y_train)
    clf = clf.fit(X_train, y_train)
    average = 'micro'
    y_pred = clf.predict_proba(X_test)[:, 1]
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, y_pred.round(), average=average)
    print('{}; Pr: {:.2f}% Rc: {:.2f}% F1: {:.2f}% Sup: {}'.format(
        average, 100 * pr, 100 * rc, 100 * f1, sup))
    print(classification_report(y_test, y_pred.round(), digits=4))
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print('Test ROC AUC: %.3f' % auc)
elif model_name == 'context_matters':
    clf = Pipeline([
        ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', LogisticRegression()),
    ])
    #print(clf.predict_proba(text)[:, 1])
    average = 'micro'
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, y_pred.round(), average=average)
    print('{}; Pr: {:.2f}% Rc: {:.2f}% F1: {:.2f}% Sup: {}'.format(
        average, 100 * pr, 100 * rc, 100 * f1, sup))
    print(classification_report(y_test, y_pred.round(), digits=4))
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print('Test ROC AUC: %.3f' % auc)
elif model_name == 'linear_models':

    print(len(X_train), len(y_train))

    model_svm = LinearSVC(random_state=31)
    print(model_svm.get_params().keys())
    params_model_svm = {'clf__penalty': ['l2'],
                        'clf__C': [10 ** i for i in np.arange(-1, 4, 0.5)],
                        }

    model_rf = RandomForestClassifier(random_state=31)
    params_model_rf = {'clf_rf__criterion': ['gini'],
                        'clf_rf__n_estimators': [1, 5, 10, 100, 300],
                        'clf_rf__min_samples_split': [2],
                        'clf_rf__min_samples_leaf': [1, 3, 5],
                        'clf_rf__max_depth': [None],
                        'clf_rf__max_features': ['auto'],
                        }

    pipeline_svm = Pipeline([
        ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', model_svm)
    ])
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf_rf', model_rf)
    ])

    params = dict(**params_model_svm)
    params_rf = dict(**params_model_rf)
    metric = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']
    # pipeline = CalibratedClassifierCV(pipeline, cv=3)
    gs_clf_svm = GridSearchCV(pipeline_svm, params, n_jobs=-1, verbose=1,
                          scoring=metric,
                          cv=3,
                          refit='f1')
    gs_clf_rf = GridSearchCV(pipeline_rf, params_rf, n_jobs=-1, verbose=1,
                             scoring=metric,
                             cv=3,
                             refit='f1')

    gs_clf_svm.fit(X_train, y_train)
    gs_clf_rf.fit(X_train, y_train)
    print('best score ({}): {}'.format(metric, gs_clf_svm.best_score_))
    print('best score ({}): {}'.format(metric, gs_clf_rf.best_score_))
    pred_svm = gs_clf_svm.best_estimator_.predict(X_test)
    pred_rf = gs_clf_rf.best_estimator_.predict(X_test)


    average = 'micro'
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, pred_svm, average=average)
    print('{}; Pr: {:.2f}% Rc: {:.2f}% F1: {:.2f}% Sup: {}'.format(
        average, 100 * pr, 100 * rc, 100 * f1, sup))
    print(classification_report(y_test, pred_svm, digits=4))
    auc = roc_auc_score(y_test, pred_svm)
    print('Test ROC AUC: %.3f' % auc)

    #RF results
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, pred_rf, average=average)
    print('{}; Pr: {:.2f}% Rc: {:.2f}% F1: {:.2f}% Sup: {}'.format(
        average, 100 * pr, 100 * rc, 100 * f1, sup))
    print(classification_report(y_test, pred_rf, digits=4))
    auc = roc_auc_score(y_test, pred_rf)
    print('Test ROC AUC: %.3f' % auc)
elif model_name == 'github':
    #find length
    print(X_train.shape, y_train.shape)
    text_col = ['comments']
    num_col = ['polarity', 'subjectivity', 'num_mention', 'nltk_score', 'perspective_score', 'stanford_polite']
    pipeline_svm = Pipeline([
        ('selector', TextSelector(column='comments')),
        ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
    ])
    pipeline_svm.fit_transform(X_train)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    transformers_list = Pipeline([('length', FunctionTransformer(get_text_length, validate=False)),
                         ('polarity', ColumnSelector(column='polarity'))])
    ''',
                         ('perspective_score', ColumnSelector(column='perspective_score')),
                         ('polite', ColumnSelector(column='stanford_polite')),
                         ('subjectivity', ColumnSelector(column='subjectivity')),
                         ('angry_word', ColumnSelector(column='num_mention')),
                         ('nltk', ColumnSelector(column='nltk_score'))
                         ])
                         '''
    print(X_train.shape)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_col),
            ('text', pipeline_svm, text_col)])

    #transformers_list.fit_transform(X_train)

    features = FeatureUnion([('text', pipeline_svm),
                            ('numerics', transformers_list)], n_jobs=-1)

    model_svm = LinearSVC(random_state=31)
    print(model_svm.get_params().keys())
    params_model_svm = {'clf__penalty': ['l2'],
                        'clf__C': [10 ** i for i in np.arange(-1, 4, 0.5)],
                        }
    pipeline = Pipeline([('features', preprocessor), ('clf', model_svm)])
    params = dict(**params_model_svm)
    metric = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']
    # pipeline = CalibratedClassifierCV(pipeline, cv=3)
    gs_clf_svm = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1,
                              scoring=metric,
                              cv=3,
                              refit='f1')


    print(X_train.shape, y_train.shape)
    gs_clf_svm.fit(X_train, y_train)
    print('best score ({}): {}'.format(metric, gs_clf_svm.best_score_))

    pred_svm = gs_clf_svm.best_estimator_.predict(X_test)

    average = 'micro'
    pr, rc, f1, sup = precision_recall_fscore_support(y_test, pred_svm, average=average)
    print('{}; Pr: {:.2f}% Rc: {:.2f}% F1: {:.2f}% Sup: {}'.format(
        average, 100 * pr, 100 * rc, 100 * f1, sup))
    print(classification_report(y_test, pred_svm, digits=4))
    auc = roc_auc_score(y_test, pred_svm)
    print('Test ROC AUC: %.3f' % auc)
