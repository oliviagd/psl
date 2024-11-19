import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
pd.options.mode.chained_assignment = None  # Disable warning

def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week

    return df

def filter_by_shared_occurrence(train, test, id_cols):
    # filter out occurrences not shared between train/test
    train = pd.merge(train, test[id_cols].drop_duplicates(), how="inner")
    test = pd.merge(test, train[id_cols].drop_duplicates(), how="inner")

    return train, test

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
    
    return pd.concat(smoothed_data, ignore_index=True)

def fit_and_predict(train_data, smoothed_data, test_data):
    train_data = train_data.merge(smoothed_data, on=['Date', 'Store', 'Dept'])
    train_data['Weekly_Sales'] = train_data['Smoothed_Weekly_Sales']
    train_data.drop(columns=['Smoothed_Weekly_Sales'], inplace=True)
    
    predictions = []
    
    departments = train_data['Dept'].unique()
    for dept in departments:
        dept_train = train_data[train_data['Dept'] == dept]
        dept_test = test_data[test_data['Dept'] == dept]
        
        X_train = dept_train[['Store', 'Year', 'Week']]
        y_train = dept_train['Weekly_Sales']
        
        X_test = dept_test[['Store', 'Year', 'Week']]
        
        #poly = PolynomialFeatures(degree=2, include_bias=False)
        #X_train = poly.fit_transform(X_train)
        #X_test = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        dept_test.loc[:, 'Weekly_Sales_Pred'] = model.predict(X_test)
        predictions.append(dept_test)
    
    return pd.concat(predictions, ignore_index=True)

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

    train, test = filter_by_shared_occurrence(train, test, ["Store", "Dept"])

    smoothed_train = svd_dept(train)
    preds = fit_and_predict(train, smoothed_train, test)
    
    preds["weight"] = preds["IsHoliday"].apply(lambda x: 5 if x else 1)

    print(
        mean_absolute_error(
            preds["Weekly_Sales"],
            preds["Weekly_Sales_Pred"],
            sample_weight=preds["weight"],
        )
    )
