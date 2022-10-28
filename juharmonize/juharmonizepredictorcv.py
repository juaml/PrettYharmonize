import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any

from sklearn.model_selection import KFold

from . import JuHarmonizePredictor
from .utils import subset_data


class JuHarmonizePredictorCV:
    def __init__(
        self, 
        n_splits: int = 10,
        random_state: Optional[int] = None,
        predictor_params: Optional[Dict['str', Any]] = None
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self._model = None
        if predictor_params is None:
            predictor_params = {}
        self.predictor_params = predictor_params

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
        extra_vars: Optional[npt.NDArray] = None,

    ) -> "JuHarmonizePredictorCV":
        self._model = JuHarmonizePredictor(**self.predictor_params)
        self._model.fit(X, y, sites, covars, extra_vars)
        return self

    def fit_transform(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
        extra_vars: Optional[npt.NDArray] = None,

    ) -> npt.NDArray:
        self._model = JuHarmonizePredictor(**self.predictor_params)
        Xout = np.empty_like(X)
        Xout[:] = np.nan
        kf = KFold(
            n_splits=self.n_splits, random_state=self.random_state, 
            shuffle=True)
        for _, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, sites_train, y_train, covars_train, extra_vars_train = \
                subset_data(train_index, X, sites, y, covars, extra_vars)

            X_test, _, _, _, _ = subset_data(test_index, X, sites, y, covars)

            # Harmonize using prediction
            self._model.fit(
                X_train, y_train, sites_train, covars_train, extra_vars_train)
            Xout[test_index, :] = self._model.transform(X_test)

        self.fit(X, y, sites, covars, extra_vars)
        return Xout

    def transform(
        self,
        X: npt.NDArray,
        extra_vars: Optional[npt.NDArray] = None
    ) -> npt.NDArray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.transform(X=X, extra_vars=extra_vars)
