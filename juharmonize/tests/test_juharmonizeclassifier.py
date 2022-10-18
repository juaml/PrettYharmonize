# %%
# Imports
import numpy as np
from numpy.testing import assert_array_equal
# import pandas as pd
from seaborn import load_dataset
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold
# from sklearn.utils import shuffle
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from juharmonize import JuHarmonizeClassifier


def test_JuHarmonizeClassifier() -> None:
    """Test JuHarmonizeClassifier Class"""
    # Load data
    df_iris = load_dataset('iris')

    # Binary clasification problem
    df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

    # Get target
    target = df_iris['species'].isin(['versicolor']).astype(int)
    target = target.to_numpy()

    # Data must be a numpy array [N_samples x N_Features]
    data = df_iris[['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']].values

    # Get samples and features form the data
    num_samples, _ = data.shape

    # generate random sites
    sites = np.random.randint(low=0, high=2, size=num_samples)


    X_train, X_test, y_train, y_test, sites_train, sites_test = \
        train_test_split(data, target, sites, train_size=0.8)


    juharm_model = JuHarmonizeClassifier()
    juharm_model.fit(X_train, y_train, sites_train)

    y_pred = juharm_model.predict(X_test, sites=sites_test)

    assert y_pred.shape == y_test.shape

    # test using string in sites
    sites = np.random.choice(['site1', 'site2'], size=num_samples)

    X_train, X_test, y_train, y_test, sites_train, sites_test = \
        train_test_split(data, target, sites, train_size=0.8)

    juharm_model = JuHarmonizeClassifier()

    juharm_model.fit(X_train, y_train, sites_train)

    y_pred = juharm_model.predict(X_test, sites=sites_test)

    assert y_pred.shape == y_test.shape
