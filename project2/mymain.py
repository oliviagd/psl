from sklearn.decomposition import TruncatedSVD
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


def svd_dept(data, n_components=5):
    """
    Apply SVD for dimensionality reduction by Department.
    """
    svd_results = []
    for dept, group in data.groupby("Dept"):
        weekly_sales_pivot = group.pivot(
            index="Date", columns="Store", values="Weekly_Sales"
        ).fillna(0)
        svd = TruncatedSVD(n_components=n_components)
        svd_transformed = svd.fit_transform(weekly_sales_pivot)
        svd_results.append((dept, svd_transformed))
    return svd_results


def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week

    return df


if __name__ == "__main__":

    results = pd.read_csv(
        "/Users/oliviadalglish/UIUC/tmp/psl/project2/Proj2_Data/test_with_label.csv"
    )
    results["Date"] = pd.to_datetime(results["Date"])

    train = pd.read_csv(f"train.csv")
    test = pd.read_csv(f"test.csv")
    weights = np.where(train["IsHoliday"] == True, 5, 1)

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

    data_by_store = []

    for store in filtered_df.Store.unique():
        df_tmp = filtered_df[filtered_df["Store"] == store]
        dummies = pd.get_dummies(df_tmp["Dept"], prefix="Dept", dtype=int)
        df_tmp_w_categorical = pd.concat([df_tmp, dummies], axis=1)
        train_tmp = df_tmp_w_categorical[df_tmp_w_categorical["istrain"] == 1].drop(
            columns=["istrain"]
        )
        test_tmp = df_tmp_w_categorical[df_tmp_w_categorical["istrain"] == 0].drop(
            columns=["istrain", "Weekly_Sales"]
        )
        # print(f"test shape: {test_tmp.shape}")

        data_by_store.append((train_tmp, test_tmp))

    errors, preds = [], []
    for store_train, store_test in data_by_store:
        X = store_train.drop(columns=["Weekly_Sales"])
        y = store_train["Weekly_Sales"]

        # fit model
        model = LinearRegression()
        model.fit(X.drop(columns=["Dept", "Date"]), y)

        # predict and calculate error
        weights = np.where(store_test["IsHoliday"] == True, 5, 1)
        y_pred = model.predict(store_test.drop(columns=["Dept", "Date"]))
        store_test["Weekly_Pred"] = y_pred

        # append store-predictions
        store_preds = store_test[["Store", "Dept", "Date", "IsHoliday", "Weekly_Pred"]]
        preds.append(store_preds)
        y_true = pd.merge(
            results, store_preds, on=["Store", "Dept", "Date", "IsHoliday"]
        ).Weekly_Sales

        wmase = mean_absolute_error(y_pred, y_true, sample_weight=weights)

        errors.append((store, wmase))

    pd.concat(preds).to_csv("mypred.csv", index=False)
    total_error = sum([error for store, error in errors])
    print(f"WMASE: {total_error}")
