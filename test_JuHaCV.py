# %%
### Imports
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from JuHa import JuHa, JuHaCV
from sklearn.datasets import load_breast_cancer

### Load data
dataset = load_breast_cancer()
target = dataset['target']
data = dataset['data']
num_samples, num_features = data.shape
# generate random sites
sites = np.random.randint(low=0,high=2,size=num_samples)

data[sites == 1] = data[sites == 1] + 1

H = JuHa()
HCV = JuHaCV()

### test Harmonization in CV
n_splits = 5
kf = KFold(n_splits=n_splits,shuffle=True,random_state=42)

Hall = H.fit(data, sites, target)
# Outer loop 
for i_fold, (train_index, test_index) in enumerate(kf.split(data)):
    HCV = HCV.fit(data, sites, target, index=train_index)
    pred_class = HCV.transform(data, sites, index=test_index)

    acc = accuracy_score(pred_class, target[test_index])
    print(f"Acc in fold {i_fold}     : {acc}")

    # leaky way 1
    harm_data_leak = Hall.transform(data, sites, target, index=test_index)
    pred_class_leak = HCV.model_pred.predict(harm_data_leak)
    acc_leak = accuracy_score(pred_class_leak, target[test_index])
    print(f"Acc leak in fold {i_fold}: {acc_leak}")

    # leaky way 2
    H = H.fit(data, sites, target, index=train_index)
    harm_data_leak = H.transform(data, sites, target, index=test_index)
    pred_class_leak = HCV.model_pred.predict(harm_data_leak)
    acc_leak = accuracy_score(pred_class_leak, target[test_index])
    print(f"Acc leak in fold {i_fold}: {acc_leak}")
