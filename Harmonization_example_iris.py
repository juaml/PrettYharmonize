# %%
### Imports
import neuroHarmonize as nh
import numpy as np
import pandas as pd
from seaborn import load_dataset
from sklearn.svm import  SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

### Variables
# k fold splits for out-loop
n_splits_out = 2
# k fold splits for inner-loop
n_splits_in = 2

#random state
random_state = 24

### Fixed variables and initializations
# names
covariate_names = ["SITE", "Class"]
# classif
clf = SVC(probability=True, random_state=random_state)

# kfold objects
kf_out = KFold(n_splits=n_splits_out,shuffle= True,random_state=random_state)
kf_in = KFold(n_splits=n_splits_in,shuffle= True,random_state=random_state)

# from svm prob to target
logit = LogisticRegression()

### Load data
df_iris = load_dataset('iris')

# df_iris = shuffle(df_iris)

# Binari clasification problem
df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

# Get target
target = df_iris['species'].isin(['versicolor']).astype(int)

# Data must be a numpy array [N_samples x N_Features]
data = df_iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Get samples and features form the data
samples, features = data.shape

# generate random sites
sites_name = np.random.randint(low=0,high=2,size=samples)

# Crate covariate matrix
covariate_matrix = pd.DataFrame(target)
covariate_matrix["SITE"] = sites_name
covariate_matrix.rename(columns={"species": "Class"}, inplace=True)
 
# Get Classes
classes = pd.unique(covariate_matrix["Class"])
# Get number of classe
n_classes = len(classes)

### Harmonization loop
# Outer loop 
for i_fold, (train_index, test_index) in enumerate(kf_out.split(data)):
    
    # Train and validation data
    data_train_all = data[train_index]
    cov_mat_train_all = covariate_matrix.iloc[train_index].reset_index(drop=True)
    
    # Test data
    data_test = data[test_index]
    cov_mat_test = covariate_matrix.iloc[test_index].reset_index(drop=True)

    # Inner loop
    cv_predictions = np.ones((train_index.shape[0], n_classes)) * -1
    for inner_train_index, inner_val_index in kf_in.split(data_train_all):

        # Train data
        data_train_inner = data_train_all[inner_train_index]
        cov_mat_train_inner = cov_mat_train_all.iloc[inner_train_index].reset_index(drop=True)

        # Validation data
        data_val_inner = data_train_all[inner_val_index]
        cov_mat_val_inner = cov_mat_train_all.iloc[inner_val_index].reset_index(drop=True)
        
        # Harmonize train data
        inner_harm_model, h_train_data = nh.harmonizationLearn(data_train_inner, cov_mat_train_inner)
        
        # Get target from the cov_matriz
        train_target_inner = cov_mat_train_inner["Class"].values.astype(int)

        # Fit the clf in the Harmonize data
        clf.fit(h_train_data, train_target_inner)
     
        # Create a copy to change the class in the harmonization
        cov_mat_val_aux = cov_mat_val_inner.copy()
        
        
        for i_cls, t_class in enumerate(classes):
            # supouse all data is the same class
            cov_mat_val_aux["Class"] = t_class

            # Harmonize data supposing one class
            h_data = nh.harmonizationApply(data_val_inner, cov_mat_val_aux, inner_harm_model)

            # Get prediction over the harmonize data
            pred_cls = clf.predict_proba(h_data)
            # import pdb; pdb.set_trace()
            cv_predictions[inner_val_index, i_cls] = pred_cls[:, 0]
 
    assert np.all(cv_predictions >= 0)

    # Get target from the cov_matriz
    all_trian_target = cov_mat_train_all["Class"].values.astype(int)
    test_target = cov_mat_test["Class"].values.astype(int)

    # Build a model to predict the class labels
    logit.fit(cv_predictions, all_trian_target)

    # Train an harmonization model with all available train data
    final_model , h_train_all = nh.harmonizationLearn(data_train_all,cov_mat_train_all)

    # Train a model over all the available data
    clf.fit(h_train_all,all_trian_target)

    # Create a copy to change the class in the harmonization
    cov_mat_test_aux = cov_mat_test.copy()

    # Get the predictions for the test set
    test_predictions = np.zeros((data_test.shape[0], n_classes))
    for i_class, t_class in enumerate(classes):
        # supouse all data is the same class
        cov_mat_test_aux["Class"] = t_class

        # Harmonize data supposing one class
        h_data = nh.harmonizationApply(data_test, cov_mat_test_aux, final_model)

        # Get prediction over the harmonize data
        pred_cls = clf.predict_proba(h_data)
        test_predictions[:, i_class] = pred_cls[:, 0]

    # Predict the test classes with the builded model
    pred_class = logit.predict(test_predictions)

    print("Predicted class acc in Kfold: " + str(i_fold))
    print(accuracy_score(pred_class, test_target))


    # do leaky predictions for comparison
    h_data_leak = nh.harmonizationApply(data_test, cov_mat_test, final_model)
    pred_cls_leak = clf.predict(h_data_leak)
    print("Leaked predicted class acc in Kfold: " + str(i_fold))
    print(accuracy_score(pred_cls_leak, test_target))

    # Use the predicted classes / the real ones won't be available in test time 
    # cov_mat_test_aux["Class"] = pred_class

    # Get the Harmonized data with the estimated classes
    # h_data_test = nh.harmonizationApply(data_test, cov_mat_test_aux, final_model)

# %%
plt.scatter(test_predictions[:,0], test_predictions[:,1], c=test_target)
plt.show()
# %%
