from typing import Optional, Union
from pathlib import Path
import numpy as np
import numpy.typing as npt
from sklearn.base import clone
import julearn

from .prettyharmonize import PrettYharmonize
from .modelstorage import ModelStorage
from .utils import check_harmonize_predictor_consistency, check_consistency


class PrettYharmonizePredictor:
    """Harmonization-prediction model.

    Parameters
    ----------
    model: str
        The learning algorithm to use for the prediction step. Must be a valid
        string for `julearn.estimators.get_model`
    """

    def __init__(
        self,
        model: Optional[str] = "rf",
        use_disk: bool = False,
        path: Optional[Union[str, Path]] = None,
    ) -> None:
        if isinstance(model, str):
            model = julearn.models.get_model(model, "regression")
        self.model = model
        self.use_disk = use_disk
        self.path = path
        self._models = None
        self._harm_model = None

    def _fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
        extra_vars: Optional[npt.NDArray] = None,
    ):
        check_consistency(X, sites, y, covars)

        self._harm_model = PrettYharmonize()
        Y = self._harm_model.fit_transform(X, y, sites, covars)

        check_harmonize_predictor_consistency(X, Y, extra_vars)
        self._models = ModelStorage(use_disk=self.use_disk, path=self.path)
        for i in range(X.shape[1]):
            t_data = X[:, i]
            if extra_vars is not None:
                t_data = np.c_[t_data, extra_vars]
            else:
                t_data = t_data[:, None]
            t_model = clone(self.model)
            t_model.fit(t_data, Y[:, i])  # type: ignore
            self._models.append(t_model)
        return Y

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
        extra_vars: Optional[npt.NDArray] = None,
    ):
        _ = self._fit(
            X=X, y=y, sites=sites, covars=covars, extra_vars=extra_vars
        )
        return self

    def fit_transform(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
        extra_vars: Optional[npt.NDArray] = None,
    ):
        X_harm = self._fit(
            X=X, y=y, sites=sites, covars=covars, extra_vars=extra_vars
        )
        return X_harm

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
