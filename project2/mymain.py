from sklearn.decomposition import TruncatedSVD
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week

    return df


def apply_svd_by_department(train, test, k=8):
    svd_columns = [f"svd_{i}" for i in range(k)]
    svd = TruncatedSVD(n_components=k)

    dept_cols = [col for col in X_train.columns if "Dept_" in col]

    X_train_reduced = svd.fit_transform(train[dept_cols])
    X_train_reduced = pd.DataFrame(X_train_reduced, columns=svd_columns)

    X_test_reduced = svd.transform(test[dept_cols])
    X_test_reduced = pd.DataFrame(X_test_reduced, columns=svd_columns)

    return (
        pd.concat(
            [train.drop(columns=dept_cols).reset_index(drop=True), X_train_reduced],
            axis=1,
        ),
        pd.concat(
            [test.drop(columns=dept_cols).reset_index(drop=True), X_test_reduced],
            axis=1,
        ),
    )


if __name__ == "__main__":

    results = pd.read_csv(
        "/Users/oliviadalglish/UIUC/tmp/psl/project2/Proj2_Data/test_with_label.csv"
    )
    results["Date"] = pd.to_datetime(results["Date"])

    train = pd.read_csv(f"train.csv")
    test = pd.read_csv(f"test.csv")

    train_transform = add_date_features(train)
    test_transform = add_date_features(test)

    # Filter out rows with no store/dept in both train and test
    train["istrain"] = 1
    test["istrain"] = 0
    df = pd.concat([train, test])

    valid_groups = df.groupby(["Store", "Dept"])["istrain"].transform(
        lambda x: set(x) == {0, 1}
    )
    filtered_df = df[valid_groups]

    # split up train data by store so that we can fit a model separately for
    # each store. list will hold tuples of (train, test) data for each store
    data_by_store = []

    for store in filtered_df.Store.unique():
        # filter data by store
        df_tmp = filtered_df[filtered_df["Store"] == store]

        # one-hot encode department variable
        dummies = pd.get_dummies(df_tmp["Dept"], prefix="Dept", dtype=int)
        df_tmp_w_categorical = pd.concat([df_tmp, dummies], axis=1)
        train_tmp = df_tmp_w_categorical[df_tmp_w_categorical["istrain"] == 1].drop(
            columns=["istrain"]
        )
        test_tmp = df_tmp_w_categorical[df_tmp_w_categorical["istrain"] == 0].drop(
            columns=["istrain", "Weekly_Sales"]
        )

        data_by_store.append((train_tmp, test_tmp))

    errors, preds = [], []
    for store_train, store_test in data_by_store:
        X_train = store_train.drop(columns=["Weekly_Sales"])
        y_train = store_train["Weekly_Sales"]

        X_train, X_test = apply_svd_by_department(X_train, store_test)

        # fit model
        model = LinearRegression()
        model.fit(X_train.drop(columns=["Dept", "Date"]), y_train)

        # predict and calculate error
        y_pred = model.predict(X_test.drop(columns=["Dept", "Date"]))
        X_test["Weekly_Pred"] = y_pred

        # append store-predictions
        store_preds = X_test[["Store", "Dept", "Date", "IsHoliday", "Weekly_Pred"]]
        preds.append(store_preds)
        y_true = pd.merge(
            results, store_preds, on=["Store", "Dept", "Date", "IsHoliday"]
        ).Weekly_Sales

        weights = np.where(store_preds["IsHoliday"] == True, 5, 1)
        wmase = mean_absolute_error(y_pred, y_true, sample_weight=weights)

        errors.append((store, wmase))

    pd.concat(preds).to_csv("mypred.csv", index=False)
    total_error = sum([error for store, error in errors])
    print(f"WMASE: {total_error}")
