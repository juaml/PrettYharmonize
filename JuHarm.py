# %%
### Imports
import neuroHarmonize as nh
import pandas as pd
import numpy as np

class JuHarm:
    def __init__(self, preserve_target=True):        
        self.preserve_target = preserve_target
        self.model = None
    
    def fit(self, data, sites, target=None, covars=None):
        assert data.shape[0] == len(sites)
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

    def transform(self, data, sites, target=None, covars=None):
        if self.model is None:
            raise Exception("Model not fitted")  

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

    def transform_target_pretend(self, data, sites, targets=None, covars=None):        
        if self.model is None:
            raise Exception("Model not fitted")

        if self.preserve_target is False:
            raise Exception("Model not fitted with target")

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
