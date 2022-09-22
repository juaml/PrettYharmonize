# %%
### Imports
import neuroHarmonize as nh
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import  SVC
from sklearn.linear_model import LogisticRegression

class JuHarm:
    def __init__(self, preserve_target=True):        
        self.preserve_target = preserve_target
        self.model = None
    
    def subset(self, data, sites, target=None, covars=None, index=None):
        if index is not None:
            data = data[index]
            sites = sites[index]
            if target is not None:
                target = target[index]
            if covars is not None:
                covars = covars.iloc[index]
        return data, sites, target, covars

    def fit(self, data, sites, target=None, covars=None, index=None):
        '''
        data: numpy array of shape [N_samples x N_features]
        sites: numpy array of shape [N_samples]
        target: numpy array of shape [N_samples]
        covars: pandas dataframe of shape [N_samples x N_covars]
        '''
        assert data.shape[0] == len(sites)
        if target is not None:
            assert data.shape[0] == len(target)
        
        data, sites, target, covars = self.subset(data, sites, target, covars, index)
        
        covarsDF = pd.DataFrame({"SITE": sites})
        if self.preserve_target:
            assert target is not None
            assert len(target) == data.shape[0]
            self.targets = np.sort(np.unique(target))
            covarsDF["Class"] = target
        
        self.expect_covars = False
        if covars is not None:
            assert isinstance(covars, pd.DataFrame)
            self.expect_covars = True
            self.covars = np.array(list(covars.columns))
            covarsDF = covarsDF.append(covars)        
        
        self.model, self.data = nh.harmonizationLearn(data, covarsDF)
        return self

    def transform(self, data, sites, target=None, covars=None, index=None):
        '''
        data: numpy array of shape [N_samples x N_features]
        sites: numpy array of shape [N_samples]
        target: numpy array of shape [N_samples]
        covars: pandas dataframe of shape [N_samples x N_covars]
        '''
        if self.model is None:
            raise Exception("Model not fitted")  

        data, sites, target, covars = self.subset(data, sites, target, covars, index)

        covarsDF = pd.DataFrame({"SITE": sites})
        if self.preserve_target:
            assert target is not None
            covarsDF["Class"] = target
        
        if self.expect_covars:
            if covars is None:
                raise Exception("Model expects covariates")
            assert isinstance(covars, pd.DataFrame)
            assert np.all(np.array(list(covars.columns)) == self.covars)
            covarsDF = covarsDF.append(covars)        
        
        return nh.harmonizationApply(data, covarsDF, self.model)

    def transform_target_pretend(self, data, sites, targets=None, 
                                 covars=None, index=None):
        '''
        data: numpy array of shape [N_samples x N_features]
        sites: numpy array of shape [N_samples]
        target: numpy array of shape [N_samples]
        covars: pandas dataframe of shape [N_samples x N_covars]      
        '''
        if self.model is None:
            raise Exception("Model not fitted")

        if self.preserve_target is False:
            raise Exception("Model not fitted with target")

        data, sites, target, covars = self.subset(data, sites, target=None,
                                      covars=covars, index=index)

        if targets is None:
            targets = self.targets

        covarsDF = pd.DataFrame({"SITE": sites})
        out = {}
        for target in targets:
            covarsDF["Class"] = [target] * len(sites)
        
            if self.expect_covars:
                if covars is None:
                    raise Exception("Model expects covariates")
                assert isinstance(covars, pd.DataFrame)
                assert np.all(np.array(list(covars.columns)) == self.covars)
                covarsDF = covarsDF.append(covars)        
            out[target] = nh.harmonizationApply(data, covarsDF, self.model)
        
        return out


# 
class JuHarmCV(JuHarm):
    def __init__(self, preserve_target=True, n_splits=5, random_state=None):
        super().__init__(preserve_target=preserve_target)
        self.model_harm = None
        self.model_meta = None
        self.model_pred = None
        self.n_splits = n_splits
        self.random_state = random_state
    
    def fit(self, data, sites, target, covars=None,
            model=None, model_meta=None, index=None):
        '''
        Fit the model using cross-validation
        '''
        assert data.shape[0] == len(sites)
        assert target is not None
        assert len(target) == data.shape[0]

        data, sites, target, covars = self.subset(data, sites, target, covars, index)

        self.targets = np.sort(np.unique(target))
        n_targets = len(self.targets)

        self.expect_covars = False
        if covars is not None:
            assert isinstance(covars, pd.DataFrame)
            self.expect_covars = True
            self.covars = np.array(list(covars.columns))

        if model is None:
            model = SVC(probability=True)
        self.model_pred = model

        if model_meta is None:
            model_meta = LogisticRegression()
        self.model_meta = model_meta

        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        cv_preds = np.ones((data.shape[0], n_targets)) * -1
        for i_fold, (train_index, test_index) in enumerate(kf.split(data)):
            H = JuHarm(preserve_target=self.preserve_target)
            H = H.fit(data, sites, target, covars, index=train_index)
            model.fit(H.data, target[train_index])
            data_pretend = H.transform_target_pretend(data, sites, 
                           covars=covars, index=test_index)        
            for i_cls, t_cls in enumerate(self.targets):            
                pred_cls =  model.predict_proba(data_pretend[t_cls])
                cv_preds[test_index, i_cls] = pred_cls[:, 0]
        
        self.model_meta.fit(cv_preds, target)

        self.model_harm = H.fit(data, sites, target, covars)
        self.model_pred.fit(self.model_harm.data, target)
        
        return self

    def transform(self, data, sites, covars=None, index=None):
        if self.model_pred is None:
            raise Exception("Model not fitted")
        
        if self.expect_covars:
            if covars is None:
                raise Exception("Model expects covariates")
            assert isinstance(covars, pd.DataFrame)
            assert np.all(np.array(list(covars.columns)) == self.covars)

        data, sites, target, covars = self.subset(data, sites, target=None, 
                                                    covars=covars, index=index)
        
        data_pretend = self.model_harm.transform_target_pretend(data, sites,
                                                                covars=covars)
        preds = np.ones((data.shape[0], len(self.targets))) * -1
        for i_cls, t_cls in enumerate(self.targets):            
                pred_cls =  self.model_pred.predict_proba(data_pretend[t_cls])
                preds[:, i_cls] = pred_cls[:, 0]
        pred = self.model_meta.predict(preds)
        return pred