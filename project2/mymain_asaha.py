import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import patsy


PROJ2_DATA_DIR = f"/Users/asaha/Documents/Illinois/CS 598 - Practical Statistical Learning/psl/project2/Proj2_Data"
NUM_FOLDS = 10


def run_approach_1(args):
    fold_num, train_csv, test_csv, pred_csv = args

    print(f"Running fold {fold_num + 1}")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    most_recent_date = train['Date'].max()

    # Filter and select necessary columns
    tmp_train = train[train['Date'] == most_recent_date].copy()
    tmp_train.rename(columns={'Weekly_Sales': 'Weekly_Pred'}, inplace=True)
    tmp_train = tmp_train.drop(columns=['Date', 'IsHoliday'])

    # Left join with the test data
    test_pred = test.merge(tmp_train, on=['Dept', 'Store'], how='left')

    # Fill NaN values with 0 for the Weekly_Pred column
    # test_pred['Weekly_Pred'].fillna(0, inplace=True)
    test_pred.fillna({'Weekly_Pred': 0}, inplace=True)


    # Write the output to CSV
    test_pred.to_csv(pred_csv, index=False)
    return fold_num

def myeval():
    test_with_label = pd.read_csv('test_with_label.csv')
    wae = []

    for i in range(NUM_FOLDS):
        file_path = f'fold_{i+1}/test.csv'
        test = pd.read_csv(file_path)
        test = test.drop(columns=['IsHoliday']).merge(test_with_label, on=['Date', 'Store', 'Dept'])

        file_path = f'fold_{i+1}/mypred.csv'
        test_pred = pd.read_csv(file_path)
        test_pred = test_pred.drop(columns=['IsHoliday'])

        new_test = test.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

        actuals = new_test['Weekly_Sales']
        preds = new_test['Weekly_Pred']
        weights = new_test['IsHoliday'].apply(lambda x: 5 if x else 1)
        wae.append(sum(weights * abs(actuals - preds)) / sum(weights))

    return wae


def run_all_folds(run_fn):
    # from multiprocessing import Pool
    # with Pool(num_folds) as p:
    #     p.map(run_fn, [
    #         (fold_num,
    #          os.path.join(PROJ2_DATA_DIR, f"fold_{fold_num + 1}", 'train.csv'),
    #          os.path.join(PROJ2_DATA_DIR, f"fold_{fold_num + 1}", 'test.csv'),
    #          os.path.join(PROJ2_DATA_DIR, f"fold_{fold_num + 1}", 'mypred.csv')
    #         )
    #         for fold_num in range(num_folds)
    #     ])
    for fold_num in range(NUM_FOLDS):
        run_fn((
            fold_num,
            os.path.join(PROJ2_DATA_DIR, f"fold_{fold_num + 1}", 'train.csv'),
            os.path.join(PROJ2_DATA_DIR, f"fold_{fold_num + 1}", 'test.csv'),
            os.path.join(PROJ2_DATA_DIR, f"fold_{fold_num + 1}", 'mypred.csv')
        ))

def run_all(run_fn):
    run_all_folds(run_fn)

    os.chdir(PROJ2_DATA_DIR)
    wae = myeval()
    for value in wae:
        print(f"\t{value:.3f}")
    print(f"{sum(wae) / len(wae):.3f}")


def run_approach_2(args):
    fold_num, train_csv, test_csv, pred_csv = args
    print(f"Running fold {fold_num + 1}")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # Define start and end dates based on test data
    start_last_year = pd.to_datetime(test['Date'].min()) - timedelta(days=375)
    end_last_year = pd.to_datetime(test['Date'].max()) - timedelta(days=350)

    # Filter train data based on the defined dates and compute 'Wk' column
    tmp_train = train[(train['Date'] > str(start_last_year))
                    & (train['Date'] < str(end_last_year))].copy()
    tmp_train['Date'] = pd.to_datetime(tmp_train['Date'])
    tmp_train['Wk'] = tmp_train['Date'].dt.isocalendar().week
    tmp_train.rename(columns={'Weekly_Sales': 'Weekly_Pred'}, inplace=True)
    tmp_train.drop(columns=['Date', 'IsHoliday'], inplace=True)

    # Compute 'Wk' column for test data
    test['Date'] = pd.to_datetime(test['Date'])
    test['Wk'] = test['Date'].dt.isocalendar().week

    # Left join with the tmp_train data
    test_pred = test.merge(tmp_train, on=['Dept', 'Store', 'Wk'], how='left').drop(columns=['Wk'])

    # Fill NaN values with 0 for the Weekly_Pred column
    test_pred.fillna({'Weekly_Pred': 0}, inplace=True)


    # Write the output to CSV
    test_pred.to_csv(pred_csv, index=False)

def svd_dept(train_data, num_components=8):
    smoothed_data = []
    departments = train_data['Dept'].unique()

    for dept in departments:
        dept_data = train_data[train_data['Dept'] == dept]

        pivot = dept_data.pivot(index='Date', columns='Store', values='Weekly_Sales').fillna(0)

        # center
        column_means = pivot.mean(axis=0)
        centered_data = pivot - column_means

        # svd
        U, S, VT = np.linalg.svd(centered_data, full_matrices=False)

        S_reduced = np.diag(S[:num_components])
        X_tilde = U[:, :num_components] @ S_reduced @ VT[:num_components, :]

        # add back the column means
        smoothed_matrix = X_tilde + column_means.values

        # revert to original format
        smoothed_df = pd.DataFrame(smoothed_matrix, index=pivot.index, columns=pivot.columns)
        smoothed_df = smoothed_df.stack().reset_index()
        smoothed_df.columns = ['Date', 'Store', 'Smoothed_Weekly_Sales']
        smoothed_df['Dept'] = dept

        smoothed_data.append(smoothed_df)


    df = pd.concat(smoothed_data, ignore_index=True)
    df['Store'] = df['Store'].astype(int)
    df['Dept'] = df['Dept'].astype(int)

    return train_data.merge(df, on=['Dept', 'Store', 'Date'], how='left').drop(columns=['Weekly_Sales']).rename(columns={'Smoothed_Weekly_Sales': 'Weekly_Sales'})


def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks
#    data['IsHoliday'] = data['IsHoliday'].apply(int)
    return data


def run_hint_2(args):
    fold_num, train_csv, test_csv, pred_csv = args
    print(f"Running fold {fold_num + 1}")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    train = svd_dept(train)
    # pre-allocate a pd to store the predictions
    test_pred = pd.DataFrame()

    train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

    train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
    train_split = preprocess(train_split)
    X = patsy.dmatrix('Weekly_Sales + Store + Dept + Yr  + Wk',
                    data = train_split,
                    return_type='dataframe')
    train_split = dict(tuple(X.groupby(['Store', 'Dept'])))


    test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
    test_split = preprocess(test_split)
    X = patsy.dmatrix('Store + Dept + Yr  + Wk',
                        data = test_split,
                        return_type='dataframe')
    X['Date'] = test_split['Date']
    test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    keys = list(train_split)

    for key in keys:
        X_train = train_split[key]
        X_test = test_split[key]

        Y = X_train['Weekly_Sales']
        X_train = X_train.drop(['Weekly_Sales','Store', 'Dept'], axis=1)

        cols_to_drop = X_train.columns[(X_train == 0).all()]
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        cols_to_drop = []
        for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
            col_name = X_train.columns[i]
            # Extract the current column and all previous columns
            tmp_Y = X_train.iloc[:, i].values
            tmp_X = X_train.iloc[:, :i].values

            coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
            if np.sum(residuals) < 1e-16:
                    cols_to_drop.append(col_name)

        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        if 'Yr' in X_train.columns:
            X_train['Yr^2'] = X_train['Yr'] ** 2
            X_test['Yr^2'] = X_test['Yr'] ** 2

        model = sm.OLS(Y, X_train).fit()
        mycoef = model.params.fillna(0)

        tmp_pred = X_test[['Store', 'Dept', 'Date']]
        X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)

        tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
        test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

    # Left join with the test data
    test_pred_joined = test.merge(test_pred, on=['Dept', 'Store', 'Date'], how='left')
    test_pred_joined.fillna({'Weekly_Pred': 0}, inplace=True)
    test_pred_joined.to_csv(pred_csv, index=False)


def main():
    # run_all(run_approach_1)
    # run_all(run_approach_2)
    run_all(run_hint_2)



if __name__ == '__main__':
    main()
