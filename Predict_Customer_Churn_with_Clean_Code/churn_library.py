# library doc string
"""
This library help to identify the credit card customers
that are most likely to churn.

Author Joaquín
Date: April 23

"""
import os
import logging

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, classification_report

sns.set()

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def import_data(pth):
    """
    return dataframe for the csv found at pth

    input:
        pth: a path to the csv

    output:
        df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder

    input:
        df: pandas dataframe

    output:
            df: pandas dataframe
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
        'Attrition_Flag'
    ]

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    column_list = [
        'Churn',
        'Customer_Age',
        'Marital_Status',
        'Total_Trans_Ct',
        'Correlation']

    for column in column_list:
        plt.figure(figsize=(20, 10))
        if column in ('Churn', 'Customer_Age'):
            df[column].hist()

        elif column == 'Marital_Status':
            df[column].value_counts('normalize').plot(kind='bar')

        elif column == 'Total_Trans_Ct':
            sns.displot(df[column], stat='density', kde=True)

        elif column == 'Correlation':
            df_copy = df.drop(cat_columns, axis=1)
            sns.heatmap(
                df_copy.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2)

        plt.savefig(f"./images/eda/{column}.jpg")
        plt.close()

    return df


def encode_helper(df, category_list, response):

    """
    This function help to encode the categorical variables
    """

    for category in category_list:
        category_lst = []
        category_groups = df[[category, response]].groupby(category).mean()

        for val in df[category]:
            category_lst.append(category_groups.loc[val])

        df[category + '_' + 'Churn'] = category_lst

    return df


def perform_feature_engineering(df, response):
    """
    input:
        df: pandas dataframe

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
    ]

    df = encode_helper(df, category_list, response)

    y = df['Churn']
    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder

    input:
        y_train: training response values
        y_test: tes response values
        y_train_preds_lr: training predictions for logistic regression
        y_train_preds_rf: training predictions for random forest
        y_test_preds_lr: test predictions for logistic regression
        y_test_preds_rf: test predictions for random forests

    output:
        None
    """

    # RandomForestClassifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')
    plt.close()

    # LogisticRegression
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and stores feature importance in pth

    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    """
    importances = model.best_estimator_.feature_importances_

    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))

    plt.title("Feature Importance")
    plt.ylabel("Importance")

    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(f"images/{output_pth}/Feature_Importance.png")
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = LogisticRegression(n_jobs=-1, max_iter=1000)

    # Parameters for Grid Search
    param_grid = {'n_estimators': [200, 500],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [4, 5, 100],
                  'criterion': ['gini', 'entropy']}

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # LogisticRegression
    lrc.fit(X_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    lrc_plot = RocCurveDisplay.from_estimator(
        lrc, X_test, y_test, ax=axis, alpha=0.8)
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8)

    plt.savefig(fname='./images/results/roc_curve_result.png')
    plt.close()

    # plt.show()

    # Compute and results
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            X_data=X_test,
                            output_pth='results')


if __name__ == "__main__":

    logging.info(f"Start program")
    pth_ = os.path.abspath("./data/bank_data.csv")

    raw_df = import_data(pth_)
    logging.info("import_data function executed successfully")

    eda_df = perform_eda(raw_df)
    logging.info("perform_eda function executed successfully")

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        eda_df, 'Churn')
    logging.info("perform_feature_engineering function executed successfully")

    train_models(X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test)
    logging.info("train_models function executed successfully")
