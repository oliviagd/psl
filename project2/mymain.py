from sklearn.decomposition import TruncatedSVD
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def svd_dept(data, n_components=5):
    """
    Apply SVD for dimensionality reduction by Department.
    """
    svd_results = []
    for dept, group in data.groupby('Dept'):
        weekly_sales_pivot = group.pivot(index='Date', columns='Store', values='Weekly_Sales').fillna(0)
        svd = TruncatedSVD(n_components=n_components)
        svd_transformed = svd.fit_transform(weekly_sales_pivot)
        svd_results.append((dept, svd_transformed))
    return svd_results


def add_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week

    return df

if __name__ == "__main__":

    train = pd.read_csv(f"train.csv")
    test = pd.read_csv(f"test.csv")

    weights = np.where(train["IsHoliday"] == True, 5, 1)

    train_transform = add_date_features(train)
    test_transform = add_date_features(test)

    # Filter out rows with no store/dept in both train and test
    train['istrain'] = 1
    test['istrain'] = 0
    df = pd.concat([train, test])

    valid_groups = df.groupby(['Store', 'Dept'])['istrain'].transform(lambda x: set(x) == {0, 1})
    filtered_df = df[valid_groups]

    data_by_store = []

    for store in filtered_df.Store.unique():
        df_tmp = filtered_df[filtered_df['Store'] == store]
        dummies = pd.get_dummies(df_tmp['Dept'], prefix="Dept", dtype=int)
        df_tmp_w_categorical = pd.concat([df_tmp, dummies], axis=1)
        train_tmp = df_tmp_w_categorical[df_tmp_w_categorical['istrain'] == 1].drop(columns=['istrain'])
        test_tmp = df_tmp_w_categorical[df_tmp_w_categorical['istrain'] == 0].drop(columns=['istrain', 'Weekly_Sales'])

        data_by_store.append((train_tmp, test_tmp))

    errors, preds = [], []
    for store_train, store_test in data_by_store:
        X = store_train.drop(columns=['Weekly_Sales'])
        y = store_train['Weekly_Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #weights = np.where(y_train["IsHoliday"] == True, 5, 1)

        model = LinearRegression()
        model.fit(X_train.drop(columns=['Dept', 'Date']), y_train)
        y_pred = model.predict(X_test.drop(columns=['Dept', 'Date']))
        X_test['Predicted_Sales'] = y_pred

        preds.append(X_test[['Store', 'Dept', 'Date', 'Predicted_Sales']])
        #wmase = mean_absolute_error(y_pred, y_test, sample_weight=weights)

        #errors.append((store, wmase))
    
    pd.concat(preds).to_csv("predictions.csv", index=False)
    #total_error = sum([error for store, error in errors])
    #print("WMASE: {total_error}")

