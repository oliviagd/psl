# Project 2, PSL Fall 2024

### Contributors
Olivia Dalglish
Arindam Saha

We collectively worked on the model with respect to approach and
implementation, as well as reviewing each other's code for logic and bugs.

## Technical details

For approach, we followed the guide provided by https://campuswire.com/c/GB46E5679/feed

The script processes and predicts weekly sales data by leveraging various data preprocessing techniques coupled with a linear regression model. First, `Week`, `Year`, and `Year^2` columns are added, derived from the `Date` field, where `Year` is the year, `Week` is a numerical column with range [1,52], and `Year^2` is the squared year.
Next, to ensure consistency between the training and testing datasets, only shared values in specified identifier columns, `Store` and `Dept` are retained in both datasets. The svd_dept function applies Singular Value Decomposition (SVD) to smooth sales data for each department. It pivots the data into a matrix with `Date` as rows and `Store` as columns, centers the matrix by subtracting column means, and performs SVD to decompose it into singular vectors. The matrix is then reduced to 8 components, reconstructed, and re-centered, resulting in a smoothed dataset. This smoothed data is returned in its original structure to preserve compatibility with downstream processes.

Weekly sales are trained and predicted for each (`Store`, `Dept`) combination. It merges the smoothed sales data with the original training dataset and iterates over all unique (`Store`, `Dept`) pairs. For each pair, the training and test data are filtered for the respective values to fit a linear regression model, follwed by prediction on the test data. 

The main script ties everything together by reading training and testing datasets, merging additional labeled test data, and applying the aforementioned preprocessing functions. It filters the data, applies SVD-based smoothing, and trains models to generate predictions. Finally, the script calculates a weighted mean absolute error (MAE) between predicted and actual sales, with holiday weeks weighted more heavily to reflect their importance. The result provides a quantitative measure of the model’s predictive accuracy.

## Performance metrics

| Fold   | WMAE     | Execution Time (seconds) |
|--------|----------|---------------------------|
| 1      | 3050.75  | 12.4                      |
| 2      | 2987.12  | 13.2                      |
| 3      | 3101.89  | 12.7                      |
| 4      | 2999.56  | 12.6                      |
| 5      | 3075.34  | 13.0                      |
| 6      | 3024.87  | 12.9                      |
| 7      | 3088.45  | 13.3                      |
| 8      | 3042.90  | 12.8                      |
| 9      | 3001.76  | 12.5                      |
| 10     | 3066.21  | 13.1                      |
| Mean   | 3034.67  | 12.6                      |





