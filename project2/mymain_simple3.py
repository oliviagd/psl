import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import TruncatedSVD


def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week

    return df


def apply_svd_by_department(train, test, k=8):
    id_cols = ["Store", "Dept"]
    train = pd.merge(train, test[id_cols], how="inner")
    test = pd.merge(test, train[id_cols], how="inner")

    train = pd.concat(
        [train, pd.get_dummies(train["Dept"], prefix="Dept", dtype=int)], axis=1
    )
    test = pd.concat(
        [test, pd.get_dummies(test["Dept"], prefix="Dept", dtype=int)], axis=1
    )

    svd_columns = [f"svd_{i}" for i in range(k)]
    svd = TruncatedSVD(n_components=k)

    dept_cols = [col for col in train.columns if "Dept_" in col]

    X_train_reduced = svd.fit_transform(train[dept_cols])
    X_train_reduced = pd.DataFrame(X_train_reduced, columns=svd_columns)

    X_test_reduced = svd.transform(test[dept_cols])
    X_test_reduced = pd.DataFrame(X_test_reduced, columns=svd_columns)

    train = train.drop(columns=dept_cols).reset_index(drop=True)
    test = test.drop(columns=dept_cols).reset_index(drop=True)

    return pd.concat([train, X_train_reduced], axis=1), pd.concat(
        [test, X_test_reduced], axis=1
    )


def main():
    test_with_label = pd.read_csv(
        "../test_with_label.csv"
    )

    train = pd.read_csv(f"train.csv")
    test = pd.read_csv(f"test.csv")
    test = pd.merge(
        test, test_with_label, on=["Store", "Dept", "Date", "IsHoliday"], how="left"
    )

    train = add_date_features(train)
    test = add_date_features(test)

    preds = []
    for store in train["Store"].unique():
        train_tmp = train[train["Store"] == store]
        test_tmp = test[test["Store"] == store]
        train_tmp, test_tmp = apply_svd_by_department(train_tmp, test_tmp)
        
        model = LinearRegression()
        model.fit(
            train_tmp.drop(columns=["Weekly_Sales", "Dept", "Date"]),
            train_tmp.Weekly_Sales,
        )

        # predict and calculate error
        test_tmp["Weekly_Sales_Pred"] = model.predict(
            test_tmp.drop(columns=["Weekly_Sales", "Dept", "Date"])
        )
        preds.append(test_tmp)

    all_preds = pd.concat(preds)
    all_preds["weight"] = all_preds["IsHoliday"].apply(lambda x: 5 if x else 1)

    print(
        mean_absolute_error(
            all_preds["Weekly_Sales"],
            all_preds["Weekly_Sales_Pred"],
            sample_weight=all_preds["weight"],
        )
    )

if __name__ == "__main__":
    main()
