import numpy as np
import numpy.typing as npt
from typing import Optional

from sklearn.model_selection import KFold

from . import JuHarmonizePredictor
from .utils import subset_data


class JuHarmonizePredictorCV:
    def __init__(
        self, 
        n_splits: int = 10,
        random_state: Optional[int] = None
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def fit_transform(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        harm_pred = JuHarmonizePredictor()

        Xout = np.empty_like(X)
        Xout[:] = np.nan
        kf = KFold(
            n_splits=self.n_splits, random_state=self.random_state, 
            shuffle=True)
        for _, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, sites_train, y_train, covars_train = subset_data(
                train_index, X, sites, y, covars)

            X_test, _, _, _ = subset_data(
                test_index, X, sites, y, covars)

            # Harmonize using prediction
            harm_pred.fit(X_train, y_train, sites_train, covars_train)
            Xout[test_index, :] = harm_pred.transform(X_test)

        return Xout
