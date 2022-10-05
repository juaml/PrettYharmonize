import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, average_precision_score, r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
    ConstantKernel,
)
from sklearn.ensemble import RandomForestRegressor as RFR
from juharmonize import (
    JuHarmonize,
    JuHarmonizeRegressor,
    JuHarmonizeClassifier,
)
from juharmonize.utils import subset_data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from pymatreader import read_mat
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing
from sklearn.base import clone
import termplotlib as tpl

n_high_var_feats = 5000
sites_use = np.array(["CoRR", "1000Gehirne", "eNKI", "CamCAN"])
sites_oos = np.array(["GOBS", "HCP"])

file_name = "/data/project/harmonize/data/VBM/all_datasets_matched_age_GMV_v4raw.mat"
data_struct = read_mat(file_name)

X = data_struct["GLMFlags"]["dat"]
female = data_struct["GLMFlags"]["Female"]
sites = np.array(data_struct["GLMFlags"]["Site"])
age = np.array(data_struct["GLMFlags"]["Age"])
print(f"Avatlable sites: {np.unique(sites)}")

n_obs_orig, n_feat_orig = X.shape

# y = female
# y = (age <= np.median(age)).astype(int)
y = age
print(y.shape)

Xorig = np.copy(X)
yorig = np.copy(y)
sitesorig = np.copy(sites)

problem_type = "regression"
if len(np.unique(y)) == 2:
    problem_type = "binary_classification"

print(f"Using sites: {sites_use}")
idx = np.zeros(len(sites))
for su in sites_use:
    idx = np.logical_or(idx, sites == su)
idx = np.where(idx)

X = X[idx]
y = y[idx]
sites = sites[idx]
covars = None
# covars = pd.DataFrame({'age':age[idx]})

np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)
assert not np.any(np.isnan(X))
assert not np.any(np.isinf(X))

# harmonization fails with low variance features
colvar = np.var(X, axis=0)
idxvar = np.argsort(-colvar)
idxvar = idxvar[range(0, n_high_var_feats)]
X = X[:, idxvar]
# X = X[:,np.random.choice(np.where(colvar > 1e-5)[0], 5000, replace=False)]
print(f"Data shape: {X.shape}")
print("Sites:")
usites, csites = np.unique(sites, return_counts=True)
assert len(usites) > 1
print(np.asarray((usites, csites)))
if problem_type == "binary_classification":
    print("Label counts:")
    uy, cy = np.unique(y, return_counts=True)
    print(np.asarray((uy, cy)))

print("Distribution of y:")
counts, bin_edges = np.histogram(y)
fig = tpl.figure()
fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
fig.show()

if len(usites) <= 3:
    for i in range(len(usites)):
        print(f"\nDistribution of y for {usites[i]}:")
        counts, bin_edges = np.histogram(y[sites == usites[i]])
        fig = tpl.figure()
        fig.hist(
            counts, bin_edges, orientation="horizontal", force_ascii=False
        )
        fig.show()


Xoos = None
yoos = None
if sites_oos is not None:
    print(f"OOS sites: {sites_oos}")
    idx = np.zeros(len(sitesorig))
    for su in sites_oos:
        idx = np.logical_or(idx, sitesorig == su)
    idx = np.where(idx)
    Xoos = Xorig[idx]
    Xoos = Xoos[:, idxvar]
    yoos = yorig[idx]
    sitesoos = sitesorig[idx]
    covarsoos = None

# print('\n*** SHUFFLING SITES ***\n')
# np.random.shuffle(sites)

# induce site difference
# for ii, ss in enumerate(sites):
#    if ss==usites[1]:
#        X[ii] += np.random.normal(1.0,1.0)
#    else:
#        X[ii] -= np.random.normal(-1.0,1.0)

model_full = JuHarmonize()
model_notarget = JuHarmonize(preserve_target=False)
model_harm = JuHarmonize()

params_svm = {"kernel": ("linear", "rbf"), "C": [0.01, 0.1, 1, 10, 100]}
kernel1 = RBF() + WhiteKernel()
kernel2 = DotProduct() + WhiteKernel()
params_gpr = {"kernel": [kernel1, kernel2]}
params_gpr = [
    {"alpha": [1e-5], "kernel": [RBF(l, (1e-7, 10e7)) for l in [0.1, 1, 10]]}
]

if problem_type == "binary_classification":
    cls_model = GridSearchCV(SVC(probability=True), params_svm)
    pred_model_cheat = clone(cls_model)
    pred_model_leak = clone(cls_model)
    pred_model_noleak = clone(cls_model)
    pred_model_pool = clone(cls_model)
    pred_model_notarget = clone(cls_model)
    pred_model_JuHaCV = clone(cls_model)
    stack_model_JuHaCV = "logit"
    model_harm_cv = JuHarmonizeClassifier(
        pred_model=pred_model_JuHaCV,
        n_folds=10,
        stack_model=stack_model_JuHaCV,
        use_cv_test_transforms=True,
        predict_ignore_site=True,
        random_state=11,
    )
else:
    # reg_model = GridSearchCV(GPR(n_restarts_optimizer=5,
    #            normalize_y=True), params_gpr)
    # reg_model = GridSearchCV(SVR(), params_svm)
    reg_model = SVR()
    pred_model_cheat = clone(reg_model)
    pred_model_leak = clone(reg_model)
    pred_model_pool = clone(reg_model)
    pred_model_notarget = clone(reg_model)
    pred_model_JuHaCV = clone(reg_model)
    stack_model_JuHaCV = "rf"  # 'svm' # 'linreg'

    model_harm_cv = JuHarmonizeRegressor(
        pred_model=pred_model_JuHaCV,
        n_folds=10,
        regression_points=100,
        stack_model=stack_model_JuHaCV,
        regression_search=False,
        regression_search_tol=float(2.0),
        use_cv_test_transforms=True,
        predict_ignore_site=True,
        random_state=11,
    )

# test Harmonization in CV
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# idx_2 = np.random.choice(range(X.shape[0]), int(X.shape[0]/2), replace=False)
# XX, ss, yy, cc = subset_data(idx_2, X, sites, y, covars)
# XX = model_full.fit(XX, yy, ss, cc)
# X_harm = model_full.transform(X, y, sites, covars)

X_harm = model_full.fit_transform(X, y, sites, covars)
X_harm_notarget = model_notarget.fit_transform(X, y, sites, covars)
print(
    f"Variance decreased for {np.sum(np.std(X_harm, axis=0) < np.std(X, axis=0))} features of {X.shape[1]}"
)

for i_fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, sites_train, y_train, covars_train = subset_data(
        train_index, X, sites, y, covars
    )

    X_test, sites_test, y_test, covars_test = subset_data(
        test_index, X, sites, y, covars
    )

    # especially useful with GPR
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    assert not np.any(np.isnan(X_train)) and not np.any(np.isinf(X_train))
    assert not np.any(np.isnan(X_test)) and not np.any(np.isinf(X_test))

    # Harmonize without target
    X_train_harm_notarget = model_notarget.fit_transform(
        X_train, y_train, sites_train, covars_train
    )
    X_test_harm_notarget = model_notarget.transform(
        X_test, y_test, sites_test, covars_test
    )

    # Harmonize X using Juha (leak)
    X_train_harm = model_harm.fit_transform(
        X_train, y_train, sites_train, covars_train
    )
    X_test_harm = model_harm.transform(X_test, y_test, sites_test, covars_test)

    # Harmonize X using JuhaCV (no leak)
    X_train_harm_cv = model_harm_cv.fit_transform(
        X_train, y_train, sites_train, covars_train
    )
    X_test_harm_cv = model_harm_cv.transform(X_test, sites_test, covars_test)

    # assert_array_equal(X_train_harm, X_train_harm_cv)  # data should be the same here

    # Fit model
    pred_model_cheat.fit(X_harm[train_index], y_train)
    pred_model_leak.fit(X_train_harm, y_train)
    pred_model_pool.fit(X_train, y_train)
    pred_model_notarget.fit(X_train_harm_notarget, y_train)

    print(f"Fold: {i_fold+1}/{n_splits}")
    if problem_type == "binary_classification":
        # harmonized without target
        pred_notarget = pred_model_notarget.predict_proba(
            X_test_harm_notarget
        )[:, 1]
        acc_notarget = average_precision_score(y_test, pred_notarget)
        # leaky way 1: complete data harmonization
        pred_cheat = pred_model_cheat.predict_proba(X_harm[test_index])[:, 1]
        acc_cheat = average_precision_score(y_test, pred_cheat)
        # leaky way 1: labels in test set
        pred_leak = pred_model_leak.predict_proba(X_test_harm)[:, 1]
        acc_leak = average_precision_score(y_test, pred_leak)
        # pool
        pred_pool = pred_model_pool.predict_proba(X_test)[:, 1]
        acc_pool = average_precision_score(y_test, pred_pool)
        # noleak
        pred_noleak = model_harm_cv.predict_proba(
            X_test, sites_test, covars_test
        )[:, 1]
        acc_noleak = average_precision_score(y_test, pred_noleak)
    else:
        # harmonized without target
        pred_notarget = pred_model_notarget.predict(X_test_harm_notarget)
        acc_notarget = mae(y_test, pred_notarget)
        # leaky way 1: complete data harmonization
        pred_cheat = pred_model_cheat.predict(X_harm[test_index])
        acc_cheat = mae(y_test, pred_cheat)
        # leaky way 1: labels in test set
        pred_leak = pred_model_leak.predict(X_test_harm)
        acc_leak = mae(y_test, pred_leak)
        # pool
        pred_pool = pred_model_pool.predict(X_test)
        acc_pool = mae(y_test, pred_pool)
        # noleak
        pred_noleak = model_harm_cv.predict(X_test, sites_test, covars_test)
        acc_noleak = mae(y_test, pred_noleak)

    print(f"notarget: {acc_notarget:.4f}")
    print(f"cheat   : {acc_cheat:.4f}")
    print(f"leak    : {acc_leak:.4f}")
    print(f"pool    : {acc_pool:.4f}")
    print(f"noleak  : {acc_noleak:.4f}")
    print("-----------------------")


if Xoos is not None:
    print(f"OOS sites: {sites_oos}")
    XX = model_harm_cv.transform(Xoos, sitesoos, covarsoos)
    acc_noleak = mae(yoos, model_harm_cv._pred_y)
    pred_pool = pred_model_pool.predict(Xoos)
    acc_pool = mae(yoos, pred_pool)
    print(f"OOS performance: noleak={acc_noleak}, pool={acc_pool}")
