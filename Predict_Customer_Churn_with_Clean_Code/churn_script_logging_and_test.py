""""
Test module for the churn_library pipeline.

Author: Joaquín
Date: April 2023

"""

import os
import logging
import pandas as pd
from math import ceil


import joblib
import pytest

from churn_library import (
    import_data,
    perform_eda,
    encode_helper,
    perform_feature_engineering,
    train_models
)

def test_import_data():
    """
    Load the raw data in csv and return a dataframe

    """
    try:
        raw_df = import_data("./data/bank_data.csv")

        assert raw_df.shape[0] > 0
        assert raw_df.shape[1] > 0
        assert type(raw_df) == pd.DataFrame

        logging.info(f"test_import_data function executed successfully")

    except Exception as error:
        raise error

def test_perform_eda():
    """
    Performe eda over raw_dataframe
    """
    try:
        raw_df = import_data("./data/bank_data.csv")
        perform_eda(raw_df)

        column_list = ['Churn', 'Customer_Age','Marital_Status','Total_Trans_Ct','Correlation']

        for column in column_list:
            assert os.path.isfile(f"./images/eda/{column}.jpg") is True
      
    except Exception as error:
        raise error
    
def test_encode_helper():
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',            
    ]
    
    try:
        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
  
        encode_df = encode_helper(df, category_list, 'Churn')

        for category in category_list:
            assert category +'_'+ 'Churn' in encode_df.columns

    except Exception as error:
        raise error


def test_perform_feature_engineering():
    """
    Test perform feature engineering

    """
    try:
        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val=="Existing Customer" else 1)
        
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

        assert X_test.shape[0] == ceil(df.shape[0]*0.3)

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]


    except Exception as error:
        raise error
    
def test_train_models():
    """
    Test train models
    """
    
    df = import_data("./data/bank_data.csv")

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val=="Existing Customer" else 1)
    
     # Feature engineering 
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    # Assert if `.pkl` file is present
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        assert os.path.isfile("./models/rfc_model.pkl") is True
    except AssertionError as err:        
        raise err

    image_list = ['Feature_Importance','logistic_results','rf_results','roc_curve_results']

    for image in image_list:
        assert os.path.isfile(f"./images/results/{image}.png") is True