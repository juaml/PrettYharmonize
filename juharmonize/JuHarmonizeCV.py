# %%
# Imports
import numpy as np
import numpy.typing as npt
from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold
from typing import Optional
import julearn

from . import JuHarmonize
from .utils import subset_data, check_consistency, check_harmonization_results
from warnings import warn


class JuHarmonizeCV:
    """Do JuHa in a CV consistent manner.

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
    regression_points: list or int
        The points to use for the regression model. If an int is given, the
        points will be chosen uniformly between the min and max of the target.
        If None, equals to 10 points (default).
    regression_search: bool
        If True, the regression points will be searched for using binary search
        TODO: @kaurao, please add more details here
    regression_search_tol: float
        The tolerance to use for the binary search. If None, equals to 1e-3.

    """

    def __init__(
        self,
        problem_type: str,
        preserve_target: bool = True,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        stack_model: Optional[str] = None,
        pred_model: Optional[str] = None,
        use_cv_test_transforms: bool = False,
        regression_points: Optional[list] = None,
        regression_search: Optional[bool] = False,
        regression_search_tol: float = 0,
    ) -> None:
        """Initialize the class."""
        assert problem_type in ["binary_classification", "regression"]

        if problem_type == 'regression' and regression_search:
            assert regression_search_tol is not None
            assert isinstance(regression_search_tol, float)
            assert regression_search_tol > 0

        self.problem_type = problem_type
        self._nh_model = None
        self.n_folds = n_folds
        self.random_state = random_state
        self.preserve_target = preserve_target
        self.use_cv_test_transforms = use_cv_test_transforms
        self.regression_points = regression_points
        self.regression_search = regression_search
        self.regression_search_tol = regression_search_tol

        if pred_model is None:
            pred_model = "svm"

        if stack_model is None:
            if problem_type == "binary_classification":
                stack_model = "logit"
            elif problem_type == "regression":
                stack_model = "gauss"

        if isinstance(stack_model, str):
            _, stack_model = julearn.api.prepare_model(
                stack_model, problem_type)

        if isinstance(pred_model, str):
            _, pred_model = julearn.api.prepare_model(
                pred_model, problem_type)

        self.pred_model: ClassifierMixin = pred_model  # type: ignore
        self.stack_model: ClassifierMixin = stack_model  # type: ignore

    def _fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ):
        """Learn the leak-free harmonization model and return the harmonized X

        Parameters
        ----------
        X: numpy array of shape [N_samples x N_features]
            the data to learn the harmonization model from
        y: numpy array of shape [N_samples, 1]
            the target variable
        sites: numpy array of shape [N_samples]
            the site of the each sample
        covars: numpy array of shape [N_samples, N_covars]
            the covariates to preserve from each sample

        Returns
        -------
        X_harmonized: np.array of shape [N_samples, N_features]
            the harmonized data
        """
        check_consistency(X, sites, y, covars, need_y=True)

        if self.problem_type == "binary_classification":
            self._classes = np.sort(np.unique(y))
            assert len(self._classes) == 2
        else:
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

        cv = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        # collect predictions over the whole data
        n_classes = len(self._classes)
        if self.problem_type == "regression" and self.regression_search:
            n_classes = 1

        # Initialize the models and results variables
        self._nh_model = JuHarmonize(preserve_target=self.preserve_target)

        if self.use_cv_test_transforms:
            X_cv_harmonized = np.zeros(X.shape)

        cv_preds = np.ones((X.shape[0], n_classes)) * -1
        for _, (train_index, test_index) in enumerate(cv.split(X)):
            X_train, sites_train, y_train, covars_train = subset_data(
                train_index, X, sites, y, covars)

            X_test, sites_test, y_test, covars_test = subset_data(
                test_index, X, sites, y, covars)

            # Learn how to harmonize the train data
            t_X_harmonized = self._nh_model.fit_transform(
                X_train, y_train, sites_train, covars_train)  # type: ignore
            check_harmonization_results(
                X_train, t_X_harmonized, sites_train, y_train)

            # Learn how to predict y from the harmonized train data
            self.pred_model.fit(t_X_harmonized, y_train)  # type: ignore

            # Get predictions in CV
            preds = self._get_predictions(X_test, sites_test, covars_test)
            cv_preds[test_index, :] = preds

            if self.use_cv_test_transforms:
                t_cv_harm = self._nh_model.transform(
                    X_test, y_test, sites_test, covars_test)  # type: ignore
                X_cv_harmonized[test_index, :] = t_cv_harm  # type: ignore

        # Train the harmonization model on all the data
        X_harmonized = self._nh_model.fit_transform(X, y, sites, covars)
        if self.use_cv_test_transforms:
            X_harmonized = X_cv_harmonized  # type: ignore

        # Train the prediction model on all the harmonized data
        self.pred_model.fit(X_harmonized, y)  # type: ignore

        # train the stack model that uses the predictions from CV
        self.stack_model.fit(cv_preds, y)  # type: ignore

        return X_harmonized

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> "JuHarmonizeCV":
        """Learn the leak-free harmonization model

        Parameters
        ----------
        X: numpy array of shape [N_samples x N_features]
            the data to learn the harmonization model from
        y: numpy array of shape [N_samples, 1]
            the target variable
        sites: numpy array of shape [N_samples]
            the site of the each sample
        covars: numpy array of shape [N_samples, N_covars]
            the covariates to preserve from each sample

        Returns
        -------
        self: JuHaCV
            the fitted model
        """
        self._fit(X, y, sites, covars)
        return self

    def fit_transform(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """Learn the leak-free harmonization model and apply it to the data.
        Note that this is not the same as `transform` because the model is
        learned on the data that it transforms. So the transformed data is
        harmonized using the Y variable.

        Parameters
        ----------
        X: numpy array of shape [N_samples x N_features]
            the data to learn the harmonization model from
        y: numpy array of shape [N_samples, 1]
            the target variable
        sites: numpy array of shape [N_samples]
            the site of the each sample
        covars: numpy array of shape [N_samples, N_covars]
            the covariates to preserve from each sample

        Returns
        -------
        X_harmonized: numpy array of shape [N_samples x N_features]
            the harmonized data
        """
        X_harmonized = self._fit(X, y, sites, covars)
        return X_harmonized

    def transform(
        self,
        X: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """Apply the leak-free harmonization model

        Parameters
        ----------
        X: numpy array of shape [N_samples x N_features]
            the data to learn the harmonization model from
        sites: numpy array of shape [N_samples]
            the site of the each sample
        covars: numpy array of shape [N_samples, N_covars]
            the covariates to preserve from each sample

        Returns
        -------
        X_harmonized: numpy array of shape [N_samples x N_features]
            the harmonized data
        """
        if self._nh_model is None:
            raise RuntimeError("Model not fitted")

        check_consistency(X, sites, covars=covars)

        preds = self._get_predictions(X, sites, covars)

        pred_y = self.stack_model.predict(preds)  # type: ignore
        self._pred_y = pred_y
        X_harmonized = self._nh_model.transform(X, pred_y, sites, covars)
        return X_harmonized

    def _get_predictions(self, X, sites, covars):
        if self.regression_search and self.problem_type == 'regression':
            preds = self._predict_search(X, sites, covars)
        else:
            preds = self._predict_classes(X, sites, covars)
        return preds

    def _predict_classes(self, X, sites, covars):
        preds = np.ones((X.shape[0], len(self._classes))) * -1
        for i_cls, t_cls in enumerate(self._classes):
            t_y = np.ones(X.shape[0], dtype=int) * t_cls
            t_X_harmonized = self._nh_model.transform(  # type: ignore
                X, t_y, sites, covars)
            if self.problem_type == "binary_classification":
                preds[:, i_cls] = \
                    self.pred_model.predict_proba(   # type: ignore
                        t_X_harmonized)[:, 0]
            else:
                preds[:, i_cls] = self.pred_model.predict(  # type: ignore
                    t_X_harmonized)
        return preds

    def _predict_search(self, X, sites, covars):
        assert self.problem_type == 'regression'
        y = np.zeros(len(sites))
        preds = np.ones((X.shape[0], 1)) * -1
        for i_X in range(X.shape[0]):
            t_X, t_sites, _, t_covars = subset_data([i_X], X, sites, y, covars)
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
