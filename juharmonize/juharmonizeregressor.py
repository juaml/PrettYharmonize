# %%
# Imports
import numpy as np
import numpy.typing as npt
from typing import Optional, Union
import julearn
from warnings import warn

from .juharmonizecv import JuHarmonizeCV
from .utils import subset_data


class JuHarmonizeRegressor(JuHarmonizeCV):
    """Do JuHarmonizeCV in a CV consistent manner for regression.

    Parameters
    ----------
    preserve_target: bool
        If True, the target variable will be preserved during harmonization.
    n_fold: int
        Number of folds to use in the K-fold CV
    random_state: int
        Random state to use for the K-fold CV
    stack_model: str
        The learning algorithm to use for the stacking step. Must be a valid
        string for `julearn.estimators.get_model`
    pred_model: str
        The learning algorithm to use for the prediction step. Must be a valid
        string for `julearn.estimators.get_model`
    use_cv_test_transforms: bool
        If True, the harmonization will be done using the K-fold CV. This will
        generate an out-of-sample harmonized X that is less prone to
        overfitting.
            pred_model_params: Optional[dict] = None,
        stack_model_params: Optional[dict] = None,
    predict_ignore_site: bool
        If True, the site will be ignored when predicting the target variable.
        This is useful when the site is not available at fitting time (i.e. 
        using data from a different site only available in the test set).
        Defaults to False.
    pred_model_params: dict, optional
        Parameters to use for the prediction model. Only used when using the
        julearn API (pred_model as a string).
    stack_model_params: dict, optional
        Parameters to use for the stacking model. Only used when using the
        julearn API (stack_model as a string).
    """

    def __init__(
        self,
        preserve_target: bool = True,
        n_splits: int = 5,
        random_state: Optional[int] = None,
        stack_model: Optional[str] = None,
        pred_model: Optional[str] = None,
        use_cv_test_transforms: bool = False,
        predict_ignore_site: bool = False,
        pred_model_params: Optional[dict] = None,
        stack_model_params: Optional[dict] = None,
        regression_points: Optional[Union[list, int]] = None,
        regression_search: Optional[bool] = False,
        regression_search_tol: float = 0,
    ) -> None:
        """Initialize the class."""
        if not isinstance(pred_model, str) and pred_model_params is not None:
            raise ValueError(
                'pred_model_params can only be used with the julearn API '
                '(pred_model as a string)')

        if not isinstance(stack_model, str) and stack_model_params is not None:
            raise ValueError(
                'stack_model_params can only be used with the julearn API '
                '(stack_model as a string)')
        if pred_model_params is None:
            pred_model_params = {}
        if pred_model is None:
            pred_model = "svm"
            pred_model_params = {"probability": True}
        if stack_model_params is None:
            stack_model_params = {}
        if stack_model is None:
            stack_model = "gauss"

        if isinstance(stack_model, str):
            _, stack_model = julearn.api.prepare_model(
                stack_model, "regression")
            stack_model = julearn.api.prepare_model_params(
                stack_model_params, stack_model
            )

        if isinstance(pred_model, str):
            _, pred_model = julearn.api.prepare_model(
                pred_model, "regression")
            pred_model = julearn.api.prepare_model_params(
                pred_model_params, pred_model
            )

        super().__init__(
            pred_model=pred_model,
            stack_model=stack_model,
            preserve_target=preserve_target,
            n_splits=n_splits,
            random_state=random_state,
            use_cv_test_transforms=use_cv_test_transforms,
            predict_ignore_site=predict_ignore_site,
        )
        self.regression_points = regression_points
        self.regression_search = regression_search
        self.regression_search_tol = regression_search_tol

    def _prepare_fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> None:
        self._sites = np.sort(np.unique(sites))
        self._y_min = min(y)
        self._y_max = max(y)
        if self.regression_points is None:
            self.regression_points = np.linspace(
                self._y_min, self._y_max, 10)
        elif isinstance(self.regression_points, int):
            self.regression_points = np.linspace(
                self._y_min, self._y_max, self.regression_points)
        self._classes = self.regression_points
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        self._y_median = np.median(y)
        if np.min(y) > np.min(self.regression_points):
            warn(
                "min(y) > min(regression_points). "
                f"Minimum value of y is {np.min(y)} but "
                "minimum value of regression points is "
                f"{np.min(self.regression_points)}")
        if np.max(y) < np.max(self.regression_points):
            warn(
                "max(y) < max(regression_points) "
                f"Maximum value of y is {np.max(y)} but "
                "maximum value of regression points is "
                f"{np.max(self.regression_points)}")

    def _pred_model_predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict the target variable using the prediction model."""
        return self.pred_model.predict(X)

    def _get_predictions(self, X, sites, covars):
        if self.regression_search:
            preds = self._predict_search(X, sites, covars)
        else:
            preds = super()._get_predictions(X, sites, covars)
        return preds

    def _predict_search(self, X, sites, covars):
        y = np.zeros(len(sites))
        preds = np.ones((X.shape[0], 1)) * -1
        for i_X in range(X.shape[0]):
            t_X, t_sites, _, t_covars, _ = subset_data(
                [i_X], X, sites, y, covars)
            cur1, cur2 = np.array([self._y_min]), np.array([self._y_max])
            ntries = 0
            cur_dif = np.Inf
            d1 = np.Inf
            d2 = np.Inf
            while cur_dif > self.regression_search_tol and ntries < 20:
                t_X_harmonized = self._nh_model.transform(  # type: ignore
                    t_X, cur1, t_sites, t_covars)
                pred1 = self.pred_model.predict(t_X_harmonized)  # type: ignore
                t_X_harmonized = self._nh_model.transform(  # type: ignore
                    t_X, cur2, t_sites, t_covars)
                pred2 = self.pred_model.predict(t_X_harmonized)  # type: ignore
                d1 = abs(cur1 - pred1)
                d2 = abs(cur2 - pred2)
                if d1 < d2:
                    cur_dif = d1
                    cur2 = (cur1 + cur2) / 2
                else:
                    cur_dif = d2
                    cur1 = (cur1 + cur2) / 2
                ntries += 1
            if d1 < d2:
                preds[i_X] = pred1[0]  # type: ignore
            else:
                preds[i_X] = pred2[0]  # type: ignore
        return preds