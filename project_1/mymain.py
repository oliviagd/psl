# from os.path import join, dirname
# import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNetCV
from scipy.stats.mstats import winsorize
from xgboost import XGBRegressor


def fit_linear(X, y, alpha=1.0):
    np.random.seed(1735)
    # Got the idea to use ElasticNetCV from this post - https://campuswire.com/c/GB46E5679/feed/609
    model = ElasticNetCV()
    model.fit(X, y)
    return model


def fit_tree(X, y):
    # Following instructions here - https://campuswire.com/c/GB46E5679/feed/327
    np.random.seed(1735)
    model = XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=5000, subsample=0.5)
    model.fit(X, y)
    return model


def preprocess_train_data(train_data):
    X = train_data.drop(columns=["Sale_Price", "PID",
                                 'Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude'])
    y = train_data.Sale_Price.apply(np.log)

    # Following instructions here - https://campuswire.com/c/GB46E5679/feed/326 https://campuswire.com/c/GB46E5679/feed/328
    X['Garage_Yr_Blt'] = X['Garage_Yr_Blt'].fillna(0)

    capped_cols = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
    X[capped_cols] = winsorize(X[capped_cols].to_numpy(), limits=[0, 0.05], axis=0)
    capped_col_values = X[capped_cols].max()

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    scaler = StandardScaler()
    scaler.fit(X[num_cols])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', min_frequency=0.01)
    ohe.fit(X[cat_cols])

    X = pd.concat(
        [
            pd.DataFrame(scaler.transform(X[num_cols]), columns=num_cols),
            pd.DataFrame(ohe.transform(X[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
        ],
        axis=1
    )

    return X, y, capped_cols, capped_col_values, num_cols, cat_cols, scaler, ohe


def preprocess_test_data(test_data, capped_cols, capped_col_values, num_cols, cat_cols, scaler, ohe):
    X = test_data.drop(columns=["PID",
                                'Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude'])

    X['Garage_Yr_Blt'] = X['Garage_Yr_Blt'].fillna(0)
    X[capped_cols] = np.minimum(X[capped_cols].to_numpy(), capped_col_values)

    X = pd.concat(
        [
            pd.DataFrame(scaler.transform(X[num_cols]), columns=num_cols),
            pd.DataFrame(ohe.transform(X[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
        ],
        axis=1
    )

    return X, test_data.PID


def save_predictions(pid, predictions, filename):
    """
    Save predictions to local file
    """
    pd.DataFrame({'PID': pid, 'Sale_Price': predictions}).to_csv(filename, index=False, header=True)

def main(train_file, test_file, linear_pred_file, tree_pred_file):
    # Step 1: Preprocess the training data, then fit the two models.
    train_data = pd.read_csv(train_file)
    X_train, y_train, capped_cols, capped_col_values, num_cols, cat_cols, scaler, ohe = preprocess_train_data(train_data)
    linear_model = fit_linear(X_train, y_train)
    tree_model = fit_tree(X_train, y_train)

    # Step 2: Preprocess test data, then save predictions into two files: mysubmission1.txt and mysubmission2.txt
    test_data = pd.read_csv(test_file)
    X_test, pid = preprocess_test_data(test_data, capped_cols, capped_col_values, num_cols, cat_cols, scaler, ohe)
    linear_pred = linear_model.predict(X_test)
    tree_pred = tree_model.predict(X_test)

    save_predictions(pid, np.exp(linear_pred), linear_pred_file)
    save_predictions(pid, np.exp(tree_pred), tree_pred_file)

    # TODO: Delete before submitting
    # return compute_rmse(linear_pred_file, tree_pred_file)

# def compute_rmse(linear_pred_file, tree_pred_file):
#     actual = pd.read_csv(join(dirname(linear_pred_file), 'test_y.csv'))
#     y_actual = actual.Sale_Price.apply(np.log).to_numpy()

#     def rmse(pred_file):
#         pred = pd.read_csv(pred_file)
#         y_pred = pred.Sale_Price.apply(np.log).to_numpy()
#         assert np.all(actual.PID == pred.PID)
#         return np.sqrt(np.mean((y_pred - y_actual) ** 2))

#     return rmse(linear_pred_file), rmse(tree_pred_file)


if __name__ == "__main__":
    # TODO: Enable before submitting
    main('train.csv', 'test.csv', 'mysubmission1.txt', 'mysubmission2.txt')

    # TODO: Delete before submitting
    # for fold in range(1, 11):
    #     fold_dir = f'./project_1/proj1/fold{fold}'
    #     t1 = time.time()
    #     linear_rmse, tree_rmse = main(join(fold_dir, 'train.csv'), join(fold_dir, 'test.csv'), join(fold_dir, 'mysubmission1.txt'), join(fold_dir, 'mysubmission2.txt'))
    #     duration = time.time() - t1
    #     print(f"Fold {fold} ({fold_dir}) - rmse(linear): {linear_rmse:.4f} rmse(tree): {tree_rmse:.4f} - {duration:.2f} s")
