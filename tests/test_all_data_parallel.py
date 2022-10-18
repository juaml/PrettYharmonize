#%%
from warnings import WarningMessage
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
)
from juharmonize import (
    JuHarmonize,
    JuHarmonizeRegressor,
    JuHarmonizeClassifier,
    JuHarmonizePredictor,
)
from juharmonize.utils import subset_data
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing
import termplotlib as tpl
from skrvm import RVR

import argparse

## Running Parameters

# parse parameters
parser = argparse.ArgumentParser(description='Attributes')
parser.add_argument("--data_dir", type=str, default="",
                    help="Data store path")
parser.add_argument("--save_dir", type=str, default="",
                    help="Save results path")
parser.add_argument("--n_high_var_feats", type=int, default=100,
                    help="High variance features")
parser.add_argument("--sites_use", nargs='+', type=str, default=["All"],
                    help="Used sites")
parser.add_argument("--sites_oos", nargs='+', type=str, default=None,
                    help="Out of distribution sites")
parser.add_argument("--unify_sites", nargs='+', type=str, default=["IXI","CORR","AOMIC"],
                    help="Sites to unify")
parser.add_argument("--problem_type", type=str, default="binary_classification",
                    help="Problem type")
parser.add_argument("--random_sites", type=bool, default=False,
                    help="If randomized sites")
parser.add_argument("--covars", type=str, default=None,
                    help="If randomized sites")
parser.add_argument("--reg_type", type=str, default="svm",
                    help="Regression model: svm or GP")
parser.add_argument("--harm_n_splits", type=int, default=10,
                    help="Folds in Harmonization")
parser.add_argument("--harm_regression_points", type=int, default=100,
                    help="Regression point")
parser.add_argument("--regression_search_tol", type=float, default=2.0,
                    help="Regression tolerance")
parser.add_argument("--stack_model_rgs", type=str, default="rf",
                    help="Stack regression model")
parser.add_argument("--harmonize_mode", type=str, default="JUHA",
                    help="Harmonization Mode")
parser.add_argument("--CV_n_splits", type=int, default = 3,
                    help="Numbers of CV folds")
parser.add_argument("--random_state", type=int, default = 23,
                    help="Random State use")

params = parser.parse_args()

def unify_sites(sites,unify_sites):

    # make all names uppercase
    unify_sites = [element.upper() for element in unify_sites] ; 

    # Transform to DF to use isin function
    unify_sites = pd.DataFrame(unify_sites)

    # Unify the IXI sites
    if unify_sites.isin(["IXI"]).any()[0]:

        sites.replace(to_replace={"IXI/Guys":"IXI",'IXI/HH':"IXI",
                                    'IXI/IOP' :"IXI"}, inplace=True)
        
    # Unify the AOMIC sites
    if unify_sites.isin(["AOMIC"]).any()[0]:

        sites.replace(to_replace={"ID1000":"AOMIC",'PIOP1':
                                    "AOMIC",'PIOP2' :"AOMIC"}, inplace=True)

    # Unify the CORR sites
    if unify_sites.isin(["CORR"]).any()[0]:
        sites_use_CoRR = ['BMB_1', 'BNU_1' ,'BNU_2', 'BNU_3' ,'HNU_1',
        'IACAS' ,'IBATRT' ,'IPCAS_1' ,'IPCAS_2', 'IPCAS_3' ,'IPCAS_4',
        'IPCAS_5', 'IPCAS_6', 'IPCAS_7' ,'IPCAS_8',  'JHNU_1', 'LMU_1',
        'LMU_2', 'LMU_3','MPG_1', 'MRN_1' ,'NKI_1' ,'NYU_1', 'NYU_2',
        'SWU_1' ,'SWU_2' ,'SWU_3' ,'SWU_4' ,'UM' ,'UPSM_1',
        'UWM' ,'Utah_1', 'Utah_2', 'XHCUMS']

        for site_CoRR in sites_use_CoRR:
            sites.replace(to_replace={site_CoRR:"CoRR"}, inplace=True)

    return sites


# Paths
data_dir = params.data_dir
save_dir = params.save_dir

# Data variables
covars = params.covars
n_high_var_feats = params.n_high_var_feats

#Site variables
sites_oos = params.sites_oos
sites_use = params.sites_use

# General
CV_n_splits = params.CV_n_splits      
random_state = params.random_state
problem_type = params.problem_type        

# Regression parameters
reg_type = params.reg_type.upper()
stack_model_rgs = params.stack_model_rgs

# Harmonizaton set up
harm_n_splits = params.harm_n_splits
harmonize_mode = params.harmonize_mode.upper()
regression_search_tol = params.regression_search_tol
harm_regression_points = params.harm_regression_points

# Results initialization
resutls = pd.DataFrame()

######################### Models Set ups
##### Clasiffiers parameters
if problem_type == "binary_classification":
    params_svm = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10, 100]}
    cls_model = GridSearchCV(SVC(probability=True), params_svm)
    ##### Regession parameters
    stack_model_JuHaCV_cls = 'logit'

else:
    stack_model_JuHaCV_rgs = stack_model_rgs #'rf' # 'svm' # 'linreg'
    if reg_type == "SVM":
        params_svm = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10, 100]}
        reg_model = GridSearchCV(SVR(), params_svm)
    elif reg_type == "GP": 
        kernel1 = RBF() + WhiteKernel()
        kernel2 = DotProduct() + WhiteKernel()
        params_gpr = {'kernel':[kernel1, kernel2]}
        params_gpr = [{"alpha": [1e-5],
        "kernel": [RBF(l, (1e-7, 10e7)) for l in [0.1, 1, 10]]}]
        reg_model = GridSearchCV(GPR(n_restarts_optimizer=5,
                    normalize_y=True), params_gpr)
    elif reg_type == "RVR":
        reg_model = RVR(kernel='poly', degree=1)    

    else:
        WarningMessage("Regression model not supported")

## Cross varidation Parameters
kf = KFold(n_splits=CV_n_splits, shuffle=True, random_state=42)

# Scaler
scaler = preprocessing.StandardScaler()

######################### Data loading and preprocessing

X_final = pd.read_csv(data_dir + "/final_data/X_final.csv")
Y_final = pd.read_csv(data_dir + "/final_data/Y_final.csv")

############### Format data
# Unify sites names
sites = Y_final["site"]

sites = unify_sites(sites, params.unify_sites)

if sites_use == "All":
    sites_use = np.unique(sites)

# put variables in the right format
sites = np.array(sites)

X_data = np.float64(X_final.to_numpy())

# Set y
if problem_type == 'binary_classification':
    female = Y_final["gender"]
    female.replace(to_replace={"F":1,"M":0}, inplace=True)
    female = np.array(female) 
    y = female
else:
    age = np.round(Y_final["age"].to_numpy())
    y = age

# Check the target have at least 2 classes
if len(np.unique(y)) == 2:
    problem_type = 'binary_classification'
    
print(y.shape)

# Keep originals
Xorig = np.copy(X_data)
yorig = np.copy(y)
sitesorig = np.copy(sites)

print(f'Using sites: {sites_use}')
# Select data form used sites
idx = np.zeros(len(sites))
for su in sites_use:
    idx = np.logical_or(idx, sites == su)
idx = np.where(idx)

X = X_data[idx]
y = y[idx]
sites = sites[idx]

np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)

# Check por inf and NaN
assert not np.any(np.isnan(X))
assert not np.any(np.isinf(X))

# harmonization fails with low variance features
colvar = np.var(X, axis=0)
idxvar = np.argsort(-colvar)
idxvar = idxvar[range(0, n_high_var_feats)]
X = X[:, idxvar]

print(f'Data shape: {X.shape}')
print('Sites:')
usites, csites = np.unique(sites, return_counts=True)

# Check that at least 2 sites are used
assert(len(usites) > 1)
print(np.asarray((usites, csites)))
if problem_type == 'binary_classification':
    print('Label counts:')
    uy, cy = np.unique(y, return_counts=True)
    print(np.asarray((uy, cy)))

print('Distribution of y:')
counts, bin_edges = np.histogram(y)
fig = tpl.figure()
fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
fig.show()

if len(usites) <= 3:
    for i in range(len(usites)):
        print(f'\nDistribution of y for {usites[i]}:')
        counts, bin_edges = np.histogram(y[sites==usites[i]])
        fig = tpl.figure()
        fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
        fig.show()

# Out of Samples set up
Xoos = None
yoos = None
if sites_oos is not None:
    print(f'OOS sites: {sites_oos}')
    idx = np.zeros(len(sitesorig))
    for su in sites_oos:
        idx = np.logical_or(idx, sitesorig == su)
    idx = np.where(idx)
    Xoos = Xorig[idx]
    Xoos = Xoos[:,idxvar]
    yoos = yorig[idx]
    sitesoos = sitesorig[idx]
    covarsoos = None

if params.random_sites:
    print('\n*** SHUFFLING SITES ***\n')
    np.random.shuffle(sites)
    # induce site difference
    for ii, ss in enumerate(sites):
        if ss==usites[1]:
            X[ii] += np.random.normal(1.0,1.0)
        else:
            X[ii] -= np.random.normal(-1.0,1.0)

# Pick a harmonization model
if harmonize_mode == "CHEAT":
    model_harm = JuHarmonize()
elif harmonize_mode == "NO_TARGET":
    model_harm = JuHarmonize(preserve_target=False)
elif harmonize_mode == "PRED":
    model_harm = JuHarmonizePredictor()

# Generate the models
if problem_type == 'binary_classification':
    
    pred_model = cls_model

    if harmonize_mode == "JUHA":
        stack_model_JuHaCV = stack_model_JuHaCV_cls 

        model_harm = JuHarmonizeClassifier(
            pred_model=pred_model,
            n_splits=harm_n_splits,
            stack_model=stack_model_JuHaCV,
            use_cv_test_transforms=True,
            predict_ignore_site=False,
            random_state=11,
        )
else:

    pred_model = reg_model

    if harmonize_mode == "JUHA":
        stack_model_JuHaCV =  stack_model_JuHaCV_rgs  

        model_harm = JuHarmonizeRegressor(
                pred_model=pred_model,
                n_splits=harm_n_splits,
                regression_points=harm_regression_points,
                stack_model=stack_model_JuHaCV,
                regression_search=False,
                regression_search_tol=regression_search_tol,
                use_cv_test_transforms=True,
                predict_ignore_site=False,
                random_state=11,
            )

########################################## Harmonization process
# If harmonization is "cheat" use the whole dataset to harmonize
if harmonize_mode == "CHEAT":
    X_harm = model_harm.fit_transform(X, y, sites, covars)

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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    assert not np.any(np.isnan(X_train)) and not np.any(np.isinf(X_train))
    assert not np.any(np.isnan(X_test)) and not np.any(np.isinf(X_test))

    ######## Harmonization cases
    # do the harmonization with a model that learns to map from no harmonized to harmonized data
    # Fit the prediction model with the already harmonize data
    if harmonize_mode == "CHEAT" :
        pred_model.fit(X_harm[train_index], y_train)
        X_test_harm = model_harm.transform(X_test, y_test, sites_test, covars_test)

    elif harmonize_mode == "JUHA" :
        # No need to train the pred_model, is trained inside
        model_harm.fit(X_train, y_train, sites_train, covars_train)
        X_test_harm = model_harm.transform(X_test, sites_test, covars_test)

    elif harmonize_mode == "PRED":

        # X_train_harm = model_harm_transform.fit_transform(X_train, y_train, sites_train, covars_train)
        # Harmonize using prediction    
        model_harm.fit(X_train, y_train, sites_train, covars_train)
        X_test_harm = model_harm.transform(X_test)
        pred_model.fit(X_train_harm, y_train)

    elif harmonize_mode == "POOL":
        # No harmonization
        pred_model.fit(X_train, y_train)
        X_test_harm  = X_test

    else:
        # For No Target, leakeage and 
        # Harmonize and get harmonized data
        X_train_harm = model_harm.fit_transform(X_train, y_train, sites_train, covars_train)
        X_test_harm = model_harm.transform(X_test, y_test, sites_test, covars_test)
        # Fit model the model with the harmonized data
        pred_model.fit(X_train_harm, y_train)
    

    print(f"Fold: {i_fold+1}/{CV_n_splits}")
    if problem_type == "binary_classification":

        if harmonize_mode == "JUHA":
            # The model is already trained when transform the test data
            acc_model = average_precision_score(y_test, model_harm._pred_y)
        else:
            # Make a prediction over harmonized test data
            predictions = pred_model.predict_proba(X_test_harm)[:, 1]
            acc_model = average_precision_score(y_test, predictions)
    else:
        
        if harmonize_mode == "JUHA":            
            # The model is already trained when transform the test data
            acc_model = mae(y_test, model_harm._pred_y)
        else: 
            # Generate a regression with the harmonized testdata
            predictions = pred_model.predict(X_test_harm)
            acc_model = mae(y_test, predictions)

    print(harmonize_mode+": "+str(acc_model))
    print("-----------------------")

    # Save the results for the K fold 
    resutls[i_fold] = [acc_model]
print("Saving Kfold Results")
resutls.index =  [harmonize_mode]
resutls.to_csv(save_dir+"HM_"+harmonize_mode+"_results_CV.csv")

if Xoos is not None:

    if harmonize_mode == "JUHA" or harmonize_mode == "PRED" or harmonize_mode == "POOL":

        resutls_OOS = pd.DataFrame()
        print(f"OOS sites: {sites_oos}")
        # set up the model_harm_cv for OOS
        if harmonize_mode == "JUHA":  
            model_harm.predict_ignore_site = True
            model_harm.regression_points = 100
            model_harm.stack_model = RFR()
            model_harm.fit(X, y, sites, covars)
            XX = model_harm.transform(Xoos, sitesoos, covarsoos)
            acc = mae(yoos, model_harm._pred_y)

        elif harmonize_mode == "PRED":  

            model_harm.fit(X, y, sites, covars)
            XX = model_harm.predict(Xoos)
            pred_model.fit(X_harm, y)
            pred_pred = pred_model.predict(XX)
            acc = mae(yoos, pred_pred)

        elif harmonize_mode == "POOL":  
            pred_model.fit(X, y)
            pred_pool = pred_model.predict(Xoos)
            acc = mae(yoos, pred_pool)
    
        print(f"OOS performance: pred={acc}")
        print("Saving OOS Results")

        resutls_OOS[0] = [acc]
        resutls.index = [harmonize_mode]
        resutls_OOS.to_csv(save_dir+"HM_"+harmonize_mode+"_resutls_OOS.csv")
    else:
        Warning("This method does not support OOS datasets")
# %%
