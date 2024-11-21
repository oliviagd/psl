import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week

    return df


def main():
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

    model = LinearRegression()
    model.fit(train.drop(columns=["Weekly_Sales", "Dept", "Date"]), train.Weekly_Sales)

    # predict and calculate error
    y_pred = model.predict(test.drop(columns=["Weekly_Sales", "Dept", "Date"]))
    test["Weekly_Sales_Pred"] = y_pred
    test["weight"] = test["IsHoliday"].apply(lambda x: 5 if x else 1)

    print(
        mean_absolute_error(
            test["Weekly_Sales"],
            test["Weekly_Sales_Pred"],
            sample_weight=test["weight"],
        )
    )


if __name__ == "__main__":
    main()
    