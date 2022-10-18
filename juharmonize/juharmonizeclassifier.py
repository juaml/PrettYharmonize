# %%
# Imports
import numpy as np
import numpy.typing as npt
from typing import Optional
import julearn

from .juharmonizecv import JuHarmonizeCV
from .utils import check_consistency


class JuHarmonizeClassifier(JuHarmonizeCV):
    """Do JuHa in a CV consistent manner for classification.

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
    ) -> None:
        """Initialize the class."""
        pred_model_params = {}
        if pred_model is None:
            pred_model = "svm"
            pred_model_params = {"probability": True}
        if stack_model is None:
            stack_model = "logit"

        if isinstance(stack_model, str):
            _, stack_model = julearn.api.prepare_model(
                stack_model, "binary_classification")

        if isinstance(pred_model, str):
            _, pred_model = julearn.api.prepare_model(
                pred_model, "binary_classification")
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
            predict_ignore_site=predict_ignore_site
        )

    def _prepare_fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> None:
        self._sites = np.sort(np.unique(sites))

        self._classes = np.sort(np.unique(y))

    def _pred_model_predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict the target variable using the prediction model."""
        return self.pred_model.predict_proba(X)[:, 0]

    def predict_proba(
        self,
        X: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """Return the prediction

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
        pred_y: numpy array of shape [N_samples, 1]
            the prediction
        """
        if self._nh_model is None:
            raise RuntimeError("Model not fitted")

        check_consistency(X, sites, covars=covars)

        preds = self._get_predictions(X, sites, covars)
        pred_y = self.stack_model.predict_proba(preds)
        return pred_y