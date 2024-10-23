import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
# Suppress only the SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def load_data(train_file: str, test_file: str):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def clean_and_process_categorical_features(train, test):

    original_columns = train.columns
    # impute missing categories with the mode
    for col in original_columns:
        train.loc[:, col] = train[col].fillna(train[col].mode()[0])
        test.loc[:, col] = test[col].fillna(test[col].mode()[0])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train)
    
    encoded_train = encoder.transform(train)
    encoded_test = encoder.transform(test)
    
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(original_columns))
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(original_columns))
    return encoded_train_df, encoded_test_df

def clean_and_process_numeric_features(data):
    # impute missing values with median
    for col in data.columns:
        data.loc[:, col] = data[col].fillna(data[col].median())

    cols = data.columns
    
    # standardize the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return pd.DataFrame(data, columns = cols)
   
def add_feature_engineered_cols(data):
    
    # add some features
    data.loc[:, 'Total_SF'] = data[['First_Flr_SF','Second_Flr_SF','Total_Bsmt_SF']].copy().sum(axis=1)
    data.loc[:, 'Total_Bath'] = data.loc[:,'Full_Bath'] + (0.5 * data.loc[:,'Half_Bath'])
    data['Has_Pool'] = data['Pool_QC'].map({"No_Pool":0, 'Excellent':1, 'Good':1, 'Typical':0, 'Fair':0})
    data['Has_Quality_Kitchen'] = data['Kitchen_Qual'].map({'Excellent':1, 'Good':1, 'Typical':0, 'Fair':0, 'Poor':0})
    data['Has_Quality_Basement'] = data['Bsmt_Qual'].map({'Excellent':1, 'Good':1, 'Typical':0, 'Fair':0, 'Poor':0})

    # drop some of the cols
    data = data.drop(columns=['Bsmt_Qual', 'Kitchen_Qual', 'Pool_QC'])

    return data
    
def preprocess_data(train: pd.DataFrame, test: pd.DataFrame):

    X_train = train.drop(columns=["Sale_Price", "PID"])
    y_train = train.Sale_Price.apply(np.log)

    X_test = test.drop(columns=["PID"])

    # remove features that have high number of null data confirm with
    # nulls = data.isna().sum()
    # print(data[data > 0])
    cols_to_drop = ['BsmtFin_SF_2', 'Misc_Val', 'Mo_Sold', 'Three_season_porch',
       'Year_Sold', 'Bsmt_Half_Bath', 'Low_Qual_Fin_SF', 'Pool_Area']
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

    X_train = add_feature_engineered_cols(X_train)
    X_test = add_feature_engineered_cols(X_test)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    X_train_numeric = clean_and_process_numeric_features(X_train[numeric_cols])
    X_test_numeric = clean_and_process_numeric_features(X_test[numeric_cols])

    X_train_cat, X_test_cat = clean_and_process_categorical_features(X_train[categorical_cols], X_test[categorical_cols])

    X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
    X_test = pd.concat([X_test_numeric, X_test_cat], axis=1)

    return X_train, y_train, X_test

def fit_ridge(X_train, y_train, X_test, alpha=1.0):

    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    
    return ridge_model.predict(X_test)

def fit_tree(X_train, y_train, X_test):
    #TODO
    tree_model = DecisionTreeRegressor(X_train, y_train)
    return tree_model.predict(X_test) 

def save_predictions(predictions, filename):
    """
    Save predictions to local file
    """
    pd.DataFrame(predictions, columns=["predictions"]).to_csv(filename, index=False)

def main(train_file, test_file):
    # Step 1: Load data
    train_data, test_data = load_data(train_file, test_file)
    
    # Step 2: Preprocess the train and test data
    X_train, y_train, X_test = preprocess_data(train_data, test_data)
    
    linear_predictions = fit_ridge(X_train, y_train, X_test, alpha=1.0)
    #tree_predictions = fit_tree(X_train, y_train, X_test)

    save_predictions(linear_predictions, "mysubmission1.txt")
    #save_predictions(tree_predictions, "mysubmission2.txt")

if __name__ == "__main__":
    
    train_file = "./train.csv"
    test_file = "./test.csv"
    
    main(train_file, test_file)
