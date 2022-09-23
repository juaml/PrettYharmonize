# %%
# Imports
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from juha.JuHa import JuHa, JuHaCV, subset_data
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC


# Load data
dataset = load_breast_cancer()
y = dataset['target']
X = dataset['data']
covars = None
num_samples, num_features = X.shape
# generate random sites
#sites = np.random.randint(low=0, high=2, size=num_samples)
#data[sites == 1] = data[sites == 1] + 1
sites = np.random.choice([1, 2], num_samples)
print('Sites: ', np.unique(sites))

model_full = JuHa()
model_harm = JuHa()
model_harm_cv = JuHaCV()

pred_model_cheat = SVC()
pred_model_leak = SVC()
pred_model_noleak = SVC()

# test Harmonization in CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

X_harmonized = model_full.fit_transform(X, y, sites, covars)

for i_fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, sites_train, y_train, covars_train = subset_data(
        train_index, X, sites, y, covars
    )

    X_test, sites_test, y_test, covars_test = subset_data(
        test_index, X, sites, y, covars
    )

    X_harmonized_train = X_harmonized[train_index]
    X_harmonized_test =  X_harmonized[test_index]

    # model_harm.fit(X_train, sites_train, y_train, covars_train)
    # model_harm_cv.fit(X_train, sites_train, y_train, covars_train)

    # Harmonize X using Juha (leak)
    X_train_harm = model_harm.fit_transform(X_train, y_train, sites_train, covars_train)
    X_test_harm = model_harm.transform(X_test, y_test, sites_test, covars_test)

    # Harmonize X using JuhaCV (no leak)
    X_train_harm_cv = model_harm_cv.fit_transform(X_train, y_train, sites_train, covars_train)
    X_test_harm_cv = model_harm_cv.transform(X_test, sites_test, covars_test)

    assert_array_equal(X_train_harm, X_train_harm_cv)  # data should be the same here

    # Fit model
    pred_model_cheat.fit(X_harmonized_train, y_train)
    pred_model_leak.fit(X_train_harm, y_train)
    pred_model_noleak.fit(X_train_harm_cv, y_train)

    # leaky way 1: complete data harmonization
    pred_class_cheat = pred_model_cheat.predict(X_harmonized_test)
    acc_cheat = accuracy_score(pred_class_cheat, y_test)
    print(f"Acc with cheat in fold {i_fold}: {acc_cheat}")

    # leaky way 1: labels in test set
    pred_class_leak = pred_model_leak.predict(X_test_harm)
    acc_leak = accuracy_score(pred_class_leak, y_test)
    print(f"Acc with leak in fold {i_fold}: {acc_leak}")

    # No leak
    pred_class_noleak = pred_model_noleak.predict(X_test_harm_cv)
    acc_noleak = accuracy_score(pred_class_noleak, y_test)
    print(f"Acc no leak in fold {i_fold}: {acc_noleak}")
    print('----------------------------------------')
