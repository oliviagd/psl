import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV


def main():
    np.random.seed(1735)

    train_file = 'train.csv'
    test_file = 'test.csv'
    submission_file = 'mysubmission.csv'

    # Training
    train_df = pd.read_csv(train_file)
    X = train_df.drop(columns=['id', 'sentiment', 'review'])
    y = train_df['sentiment']

    model = LogisticRegressionCV(penalty='l2', scoring='roc_auc')
    model.fit(X, y)

    # Testing
    test_df = pd.read_csv(test_file)
    X_test = test_df.drop(columns=['id', 'review'])
    probs = model.predict_proba(X_test)[:, 1]

    # Save predictions
    pd.DataFrame({'id': test_df['id'], 'prob': probs}).to_csv(submission_file, index=False, header=True)


if __name__ == "__main__":
    main()
