# %%
# Imports
import neuroHarmonize as nh
import pandas as pd
import numpy.typing as npt
from typing import Optional

from .utils import check_consistency


class JuHarmonize:
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
    ) -> "JuHarmonize":
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
        check_consistency(X, sites, y, covars)

        df_covars = pd.DataFrame({"SITE": sites})
        if self.preserve_target:
            if y is None:
                raise ValueError(
                    "If preserve_target=True, then y must be provided"
                )

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
        check_consistency(X, sites, y, covars)
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
