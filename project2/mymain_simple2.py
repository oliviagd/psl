import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week

    return df


if __name__ == "__main__":
    test_with_label = pd.read_csv(
        "/Users/oliviadalglish/UIUC/tmp/psl/project2/Proj2_Data/test_with_label.csv"
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
