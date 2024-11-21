import time
import os
from os.path import dirname, join

import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy


def svd_dept(train_data, num_components=8):
    # Using idea from https://campuswire.com/c/GB46E5679/feed/457

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

    return train_data.merge(df, on=['Dept', 'Store', 'Date'], how='left') \
        .drop(columns=['Weekly_Sales']) \
        .rename(columns={'Smoothed_Weekly_Sales': 'Weekly_Sales'})


def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks
#    data['IsHoliday'] = data['IsHoliday'].apply(int)
    return data


def main(train_csv, test_csv, pred_csv):
    # Referenced from https://liangfgithub.github.io/Proj/F24_Proj2_hints_2_Python.html ("Efficient Implementation")

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
            # Idea from https://campuswire.com/c/GB46E5679/feed/462
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



def myeval(test_with_label_csv, test_csv, pred_csv):
    test_with_label = pd.read_csv(test_with_label_csv)

    test = pd.read_csv(test_csv)
    test = test.drop(columns=['IsHoliday']).merge(test_with_label, on=['Date', 'Store', 'Dept'])

    test_pred = pd.read_csv(pred_csv)
    test_pred = test_pred.drop(columns=['IsHoliday'])

    new_test = test.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

    actuals = new_test['Weekly_Sales']
    preds = new_test['Weekly_Pred']
    weights = new_test['IsHoliday'].apply(lambda x: 5 if x else 1)
    return sum(weights * abs(actuals - preds)) / sum(weights)


def run_all_folds():
    cwd = os.getcwd()

    proj2_data_dir = join(dirname(__file__), 'Proj2_Data')
    test_with_label_csv = join(proj2_data_dir, 'test_with_label.csv')

    num_folds = 10
    wae = []
    times = []
    for fold_num in range(num_folds):
        fold_dir = join(proj2_data_dir, f"fold_{fold_num + 1}")
        os.chdir(fold_dir)

        t1 = time.time()
        main(join(fold_dir, 'train.csv'), join(fold_dir, 'test.csv'), join(fold_dir, 'mypred.csv'))
        t2 = time.time()

        fold_wae = myeval(test_with_label_csv, join(fold_dir, 'test.csv'), join(fold_dir, 'mypred.csv'))
        print(f"Fold {fold_num + 1}: wae: {fold_wae:.3f} time: {t2 - t1:.3f} s")

        wae.append(fold_wae)
        times.append(t2 - t1)

    print(f"avg. wae: {sum(wae) / len(wae):.3f} time: {sum(times) / len(times):.3f} s")
    os.chdir(cwd)


if __name__ == '__main__':
    # TODO: Enable before submitting
    main('train.csv', 'test.csv', 'mypred.csv')

    # TODO: Delete before submitting
    # run_all_folds()
