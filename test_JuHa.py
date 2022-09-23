# %%
### Imports
import numpy as np
import pandas as pd
from seaborn import load_dataset
from sklearn.svm import  SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from JuHa import JuHa

### Load data
df_iris = load_dataset('iris')

# Binary clasification problem
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

# Get target
target = df_iris['species'].isin(['versicolor']).astype(int)
target = target.to_numpy()
# Get Classes
classes = np.unique(target)
# Get number of classe
n_classes = len(classes)

# Data must be a numpy array [N_samples x N_Features]
data = df_iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Get samples and features form the data
num_samples, num_features = data.shape

# generate random sites
sites = np.random.randint(low=0,high=2,size=num_samples)

# test Harmonization implementation
H = JuHa()
H = H.fit(data, sites, target)
X = H.transform(data, sites, target)
assert np.all(X == H.data)

### test Harmonization in CV
### Variables
# k fold splits for out-loop
n_splits_out = 2
# k fold splits for inner-loop
n_splits_in = 2

#random state
random_state = 24
clf = SVC(probability=True,random_state=random_state)

# kfold objects
kf_outer = KFold(n_splits=n_splits_out,shuffle=True,random_state=random_state)
kf_inner = KFold(n_splits=n_splits_in,shuffle=True,random_state=random_state)

# from svm prob to target
logit = LogisticRegression()
# Outer loop 
for i_fold, (outer_train_index, outer_test_index) in enumerate(kf_outer.split(data)):

    # Inner loop
    cv_predictions = np.ones((outer_train_index.shape[0], n_classes)) * -1
    for inner_train_index, inner_test_index in kf_inner.split(outer_train_index):

        # data index
        idx_train_inner = outer_train_index[inner_train_index]
        idx_test_inner = outer_train_index[inner_test_index]
        
        # Harmonize training data
        Hinner = H.fit(data[idx_train_inner], sites[idx_train_inner], target[idx_train_inner])
        
        # Fit the clf to harmonized data
        clf.fit(Hinner.data, target[idx_train_inner])
        
        # Predict the probabilities of the inner test data
        data_pretend_inner = Hinner.transform_target_pretend(data[idx_test_inner], sites[idx_test_inner])        
        for i_cls, t_cls in enumerate(classes):            
            pred_cls =  clf.predict_proba(data_pretend_inner[t_cls])
            # import pdb; pdb.set_trace()
            cv_predictions[inner_test_index, i_cls] = pred_cls[:, 0]
 
    assert np.all(cv_predictions >= 0)

    # Build a model to predict the class labels
    logit.fit(cv_predictions, target[outer_train_index])

    # Train an harmonization model with all available train data
    Houter = H.fit(data[outer_train_index], sites[outer_train_index], target[outer_train_index])

    # Train a model over all the available data
    clf.fit(Houter.data, target[outer_train_index])

    test_predictions = np.zeros((outer_test_index.shape[0], n_classes))
    data_pretend_outer = Houter.transform_target_pretend(data[outer_test_index], sites[outer_test_index])        
    for i_class, t_class in enumerate(classes):
        pred_cls =  clf.predict_proba(data_pretend_outer[t_cls])
        test_predictions[:, i_class] = pred_cls[:, 0]

    # Predict the test classes with the builded model
    pred_class = logit.predict(test_predictions)
    acc = accuracy_score(pred_class, target[outer_test_index])
    print(f"Acc in fold {i_fold}: {acc}")

    # do leaky predictions for comparison
    data_leak = Houter.transform(data[outer_test_index], sites[outer_test_index], target[outer_test_index])
    pred_cls_leak = clf.predict(data_leak)
    acc_leak = accuracy_score(pred_cls_leak, target[outer_test_index])
    print(f"Acc leaked in fold {i_fold}: {acc_leak}")
