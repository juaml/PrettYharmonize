# %%
# Imports
import neuroHarmonize as nh
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold
from typing import Optional, Tuple, Union
import julearn


def subset_data(
    index: npt.NDArray,
    X: npt.NDArray,
    sites: npt.NDArray,
    y: Optional[npt.NDArray] = None,
    covars: Optional[npt.NDArray] = None,
) -> Tuple[
    npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]
]:
    _X = X[index]
    _sites = sites[index]
    _y = None
    _covars = None
    if y is not None:
        _y = y[index]
    if covars is not None:
        _covars = covars[index]
    return _X, _sites, _y, _covars


def _check_consistency(
    X: npt.NDArray,
    sites: npt.NDArray,
    y: Optional[npt.NDArray] = None,
    covars: Optional[npt.NDArray] = None,
    need_y: bool = False,
) -> None:
    """Check that the dimensions of the data are consistent."""
    assert (
        X.shape[0] == sites.shape[0]
    ), "X and sites must have the same number of samples"

    assert need_y is False or y is not None, "y must be provided"
    if y is not None:
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same number of samples"

    if covars is not None:
        assert (
            covars.shape[0] == X.shape[0]
        ), "covars and X must have the same number of samples"


def _check_harmonization_results(
    X: npt.NDArray,
    harmonized_X: npt.NDArray,
    sites: npt.NDArray,
    y: npt.NDArray,
) -> None:
    if np.isnan(harmonized_X).any() or np.isinf(harmonized_X).any():
        print("Warning: NaNs or Infs in harmonized data")
        print(f"Sites: {np.unique(sites)}")
        print(f"Targets: {np.unique(y)}")
        data_colvar = np.var(X, axis=0)
        print(f"Data columns with low variance: {np.sum(data_colvar < 1e-6)}",)
        raise RuntimeError("Harmonization of trainig data failed in CV!")


class JuHa:
    """Class to perform JuHa harmonization (learn and apply)

    Parameters
    ----------
    preserve_target : bool, optional
        Whether to preserve the target (y) variable during harmonization. This
        means that the target will be included in the covariates during
        harmonization. The default is True.
    """

    def __init__(self, preserve_target=True) -> None:
        """Initialize the class."""
        self.preserve_target = preserve_target
        self._nh_model = None

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> "JuHa":
        """
        Learn the harmonization model

        Parameters
        ----------
        X: numpy array of shape [N_samples x N_features]
            The data to learn the harmonizatoin model from
        y: numpy array of shape [N_samples, 1]
            The target variable
        sites: numpy array of shape [N_samples]
            The site of the each sample
        covars: numpy array of shape [N_samples, N_covars]
            The covariates to preserve from each sample
        """
        _check_consistency(X, sites, y, covars)

        df_covars = pd.DataFrame({"SITE": sites})
        if self.preserve_target:
            if y is None:
                raise ValueError(
                    "If preserve_target=True, then y must be provided"
                )

            # self._classes = np.sort(np.unique(y))
            df_covars["y"] = y

        self._need_covars = False
        if covars is not None:
            _covars = pd.DataFrame(covars)
            self._need_covars = True
            self._covar_names = list(_covars.columns)
            df_covars = df_covars.append(_covars)

        self._nh_model, _ = nh.harmonizationLearn(X, df_covars)  # type: ignore
        return self

    def fit_transform(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """
        Learn the harmonization model and apply it to the data

        Parameters
        ----------
        X: numpy array of shape [N_samples x N_features]
            The data to learn the harmonizatoin model from
        y: numpy array of shape [N_samples, 1]
            The target variable
        sites: numpy array of shape [N_samples]
            The site of the each sample
        covars: numpy array of shape [N_samples, N_covars]
            The covariates to preserve from each sample

        Returns
        -------
        X_harmonized: numpy array of shape [N_samples x N_features]
            The harmonized data
        """
        self.fit(X, y, sites, covars)
        return self.transform(X, y, sites, covars)

    def transform(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """
        Apply the harmonization model to the data

        Parameters
        ----------
        X: numpy array of shape [N_samples x N_features]
            The data to learn the harmonizatoin model from
        y: numpy array of shape [N_samples, 1]
            The target variable
        sites: numpy array of shape [N_samples]
            The site of the each sample
        covars: numpy array of shape [N_samples, N_covars]
            The covariates to preserve from each sample

        Returns
        -------
        X_harmonized: numpy array of shape [N_samples x N_features]
            The harmonized data
        """
        if self._nh_model is None:
            raise RuntimeError("Model not fitted")
        _check_consistency(X, sites, y, covars)
        df_covars = pd.DataFrame({"SITE": sites})
        if self.preserve_target:
            if y is None:
                raise ValueError(
                    "Model was fitted with target information: "
                    "y must be provided"
                )
            df_covars["y"] = y

        if self._need_covars:
            if covars is None:
                raise RuntimeError("Model was fitted with covariates")
            _covars = pd.DataFrame(covars)
            if list(_covars.columns) != self._covar_names:
                raise ValueError(
                    "Covariates do not match. \n"
                    f"\t Expected covariates: {self._covar_names}\n"
                    f"\t Received covariates: {list(_covars.columns)}"
                )
            df_covars = df_covars.append(covars)

        X_harmonized = nh.harmonizationApply(X, df_covars, self._nh_model)
        return X_harmonized  # type: ignore


class JuHaCV:
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

    """

    def __init__(
        self,
        problem_type: str,
        preserve_target: bool = True,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        stack_model: Optional[str] = None,
        pred_model: Optional[str] = None,
        regression_points: Optional[list] = None,
    ) -> None:
        """Initialize the class."""
        assert problem_type in ["binary_classification", "regression"]
        
        self.problem_type = problem_type
        self._nh_model = None
        self.n_folds = n_folds
        self.random_state = random_state
        self.preserve_target = preserve_target
        self.regression_points = regression_points

        if pred_model is None:
            pred_model = "svm"
        
        if stack_model is None:
            if problem_type == "binary_classification":
                stack_model = "logit"
            elif problem_type == "regression":
                stack_model = "svm"

        if isinstance(stack_model, str):
            _, stack_model = julearn.api.prepare_model(
                stack_model, problem_type)
        
        if isinstance(pred_model, str):
            _, pred_model = julearn.api.prepare_model(
                pred_model, problem_type)

        if problem_type == "binary_classification":
            self.pred_model: ClassifierMixin = pred_model  # type: ignore
            self.stack_model: ClassifierMixin = stack_model  # type: ignore
        elif problem_type == "regression":
            self.pred_model: RegressorMixin = pred_model
            self.stack_model: RegressorMixin = stack_model
        #self.pred_model.set_params(probability=True)

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
        return_data: bool = False,
    ) -> Union[npt.NDArray, "JuHaCV"]:
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
        return_data: bool
            If True, the harmonized data will be returned
        """
        _check_consistency(X, sites, y, covars, need_y=True)

        if self.problem_type == "binary_classification":
            self._classes = np.sort(np.unique(y))
            assert len(self._classes) == 2
        else:
            if self.regression_points is None:
                self.regression_points = np.linspace(min(y), max(y), 10)
            elif isinstance(self.regression_points, int):
                n_point = self.regression_points
                self.regression_points = np.linspace(min(y), max(y), n_point)
            self._classes = self.regression_points
            self._y_min = min(y)
            self._y_max = max(y)
            self._y_mean = np.mean(y)
            self._y_std = np.std(y)
            self._y_median = np.median(y) 
            if np.min(y) > np.min(self.regression_points):
                print("Warning: min(y) > min(regression_points)"
                    f"Minimum value of y is {np.min(y)} but "
                    f"minimum value of regression points is "
                    f"{np.min(self.regression_points)}")
            if np.max(y) < np.max(self.regression_points):
                print("Warning: max(y) < max(regression_points)"
                    f"Maximum value of y is {np.max(y)} but "
                    f"maximum value of regression points is "
                    f"{np.max(self.regression_points)}")
        n_classes = len(self._classes)

        cv = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        # collect predictions over the whole data
        cv_preds = np.ones((X.shape[0], n_classes)) * -1
        for i_fold, (train_index, test_index) in enumerate(cv.split(X)):
            X_train, sites_train, y_train, covars_train = subset_data(
                train_index, X, sites, y, covars)

            X_test, sites_test, y_test, covars_test = subset_data(
                test_index, X, sites, y, covars)

            # harmonize train data
            t_model = JuHa(preserve_target=self.preserve_target)

            # Learn how to harmonize the train data
            t_X_harmonized = t_model.fit_transform(
                X_train, y_train, sites_train, covars_train)  # type: ignore
            _check_harmonization_results(X_train, t_X_harmonized,
            sites_train, y_train)

            # Learn how to predict y from the harmonized train data
            self.pred_model.fit(t_X_harmonized, y_train)  # type: ignore

            # For each class, predict the probability of the test data if
            # it was harmonized as belonging to that class
            for i_class, t_class in enumerate(self._classes):
                t_y_test = np.ones(len(y_test), dtype=int) * t_class
                X_test_harmonized = t_model.transform(
                    X_test, t_y_test, sites_test, covars_test)
                _check_harmonization_results(X_test, X_test_harmonized,
                sites_test, t_y_test)
                if self.problem_type == "binary_classification":
                    cv_preds[test_index, i_class] = \
                    self.pred_model.predict_proba(X_test_harmonized)[:, 0]
                else:
                    cv_preds[test_index, i_class] = \
                    self.pred_model.predict(X_test_harmonized)

        # Train the harmonization model on all the data
        self._nh_model = JuHa(preserve_target=self.preserve_target)
        X_harmonized = self._nh_model.fit_transform(X, y, sites, covars)

        # Train the prediction model on all the harmonized data
        self.pred_model.fit(X_harmonized, y)  # type: ignore

        # train the stack model that uses the predictions from CV
        self.stack_model.fit(cv_preds, y)  # type: ignore

        if return_data:
            return X_harmonized
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
        X_harmonized = self.fit(X, y, sites, covars, return_data=True)
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

        _check_consistency(X, sites, covars=covars)

        preds = self._transform_predict(X, sites, covars)
        
        if self.problem_type == "binary_classification":
            self.pred_y_proba = self.stack_model.predict_proba(preds)  # type: ignore
        pred_y = self.stack_model.predict(preds)  # type: ignore
        self._pred_y = pred_y
        X_harmonized = self._nh_model.transform(X, pred_y, sites, covars)
        return X_harmonized

    def _transform_predict(self, X, sites, covars):
        preds = np.ones((X.shape[0], len(self._classes))) * -1
        for i_cls, t_cls in enumerate(self._classes):
            t_y = np.ones(X.shape[0], dtype=int) * t_cls
            t_X_harmonized = self._nh_model.transform(X, t_y, sites, covars)
            if self.problem_type == "binary_classification":
                preds[:, i_cls] = self.pred_model.predict_proba(  # type: ignore
                    t_X_harmonized)[:, 0]
            else:
                preds[:, i_cls] = self.pred_model.predict(  # type: ignore
                    t_X_harmonized)
        return preds
