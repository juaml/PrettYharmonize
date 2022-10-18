from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
import julearn

from .juharmonize import JuHarmonize
from .utils import check_harmonize_predictor_consistency, check_consistency


class JuHarmonizePredictor:
    """Harmonization-prediction model.

    Parameters
    ----------
    model: str
        The learning algorithm to use for the prediction step. Must be a valid
        string for `julearn.estimators.get_model`
    """

    def __init__(self, model: Optional[str] = "rf"):
        if isinstance(model, str):
            _, model = julearn.api.prepare_model(
                model, "regression"
            )
        self.model = model
        self._models = None
        self._harm_model = None

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
        extra_vars: Optional[npt.NDArray] = None,
    ):
        check_consistency(X, sites, y, covars)

        self._harm_model = JuHarmonize()
        Y = self._harm_model.fit_transform(X, y, sites, covars)

        check_harmonize_predictor_consistency(X, Y, extra_vars)
        self._models = []
        for i in range(X.shape[1]):
            t_data = X[:, i]
            if extra_vars is not None:
                t_data = np.c_[t_data, extra_vars]
            else:
                t_data = t_data[:, None]
            t_model = clone(self.model)
            t_model.fit(t_data, Y[:, i])
            self._models.append(t_model)
        return self
    
    def transform(
        self,
        X: npt.NDArray,
        extra_vars: Optional[npt.NDArray] = None,
    ):
        check_harmonize_predictor_consistency(X, None, extra_vars)
        preds = []
        for i in range(X.shape[1]):
            t_data = X[:, i]
            if extra_vars is not None:
                t_data = np.c_[t_data, extra_vars]
            else:
                t_data = t_data[:, None]
            preds.append(self._models[i].predict(t_data))
        return np.array(preds).T
