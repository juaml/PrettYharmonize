# %%
# Imports
from cmath import log
import numpy as np
import numpy.typing as npt
from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.base import clone
from typing import Optional

from . import JuHarmonize
from .utils import subset_data, check_consistency, check_harmonization_results
from .logging import logger

from itertools import combinations

class JuHarmonizeCV:
    """Do JuHarmonize in a CV consistent manner.

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
    predict_ignore_site: bool False, 
        If True, the site will be ignored when predicting the target variable. 
        This is useful when the site is not available at test time 
        (i.e. using data from a different site only available in the test set).
    """

    def __init__(
        self,
        pred_model,  # TODO: Type
        stack_model,  # TODO: Type
        preserve_target: bool = True,
        n_splits: int = 5,
        random_state: Optional[int] = None,
        use_cv_test_transforms: bool = False,
        predict_ignore_site: bool = False,
        stack_model_features: Optional[list] = ['pred_harm_target'],
    ) -> None:
        """Initialize the class."""
        
        self.n_splits = n_splits
        self.random_state = random_state
        self.preserve_target = preserve_target
        self.use_cv_test_transforms = use_cv_test_transforms
        self.predict_ignore_site = predict_ignore_site        
        self.stack_model = stack_model
        self._classes = None
        self._sites = None
        
        assert isinstance(stack_model_features, list), "stack_model_features must be a list"
        assert len(stack_model_features) > 0, "stack_model_features must have at least one element"
        allowed_features = ['pred_harm_target', 'pred_harm_notarget', 'pred_orig', 'orig']
        assert all([f in allowed_features for f in stack_model_features]), f"stack_model_features must be in {allowed_features}"
        self.stack_model_features = stack_model_features
        
        # we can have multiple pred and nh models depending on stack_model_features
        self._nh_model = {}
        self._nh_model['model'] = None
        self.pred_model = {}
        for f in stack_model_features:
            if 'pred_' in f:
                self.pred_model[f] = pred_model
            if f == 'pred_harm_notarget':
                self._nh_model['notarget'] = None

    def _prepare_fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sites: npt.NDArray,
        covars: Optional[npt.NDArray] = None,
    ):
        raise NotImplementedError("Implement this method in the subclass")

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
        X_harmonized = None
        # get features for the stack model using CV
        logger.info(f"Fitting data ({X.shape})")
        cv = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        # # collect predictions over the whole data
        # n_classes = len(self._classes)
        
        if 'pred_harm_target' in self.stack_model_features:
            self._nh_model['model'] = JuHarmonize(preserve_target=self.preserve_target)
            if self.use_cv_test_transforms:
                X_cv_harmonized = np.zeros(X.shape)
            
        if 'pred_harm_notarget' in self.stack_model_features:
            self._nh_model['notarget'] = JuHarmonize(preserve_target=False)
            if self.use_cv_test_transforms:
                X_cv_harmonized_notarget = np.zeros(X.shape)

        # get CV based predictions for stacking
        cv_preds = None
        logger.info(f"Starting fitting CV using {cv}")
        for i_fold, (train_index, test_index) in enumerate(cv.split(X)):
            logger.info(f"\tStarting fold {i_fold}")
            X_train, sites_train, y_train, covars_train, _ = subset_data(
                train_index, X, sites, y, covars)
            X_test, sites_test, y_test, covars_test, _ = subset_data(
                test_index, X, sites, y, covars)

            # Learn how to harmonize the train data
            if 'pred_harm_target' in self.stack_model_features:
                logger.info("\tFitting neuroHarmonize model")
                t_X_harmonized = self._nh_model['model'].fit_transform(
                    X_train, y_train, sites_train, covars_train)  # type: ignore
                logger.info("\tChecking harmonization results")
                check_harmonization_results(
                    X_train, t_X_harmonized, sites_train, y_train)

                logger.info("\tFitting predictive model")
                # Learn how to predict y from the harmonized train data
                self.pred_model['pred_harm_target'].fit(t_X_harmonized, y_train)  # type: ignore
            
            if 'pred_harm_notarget' in self.stack_model_features:
                logger.info("\tFitting neuroHarmonize model")
                t_X_harmonized = self._nh_model['notarget'].fit_transform(
                    X_train, y_train, sites_train, covars_train)  # type: ignore
                logger.info("\tChecking harmonization results")
                check_harmonization_results(
                    X_train, t_X_harmonized, sites_train, y_train)

                logger.info("\tFitting predictive model")
                # Learn how to predict y from the harmonized train data
                self.pred_model['pred_harm_notarget'].fit(t_X_harmonized, y_train)
            
            if 'pred_orig' in self.stack_model_features:
                self.pred_model['pred_orig'].fit(X_train, y_train)
                        
            # Get predictions in CV
            logger.info("\tGetting predictions")
            preds = self._get_predictions(X_test, sites_test, covars_test)
            if cv_preds is None:
                cv_preds = np.ones((X.shape[0], preds.shape[1])) * -1
            cv_preds[test_index, :] = preds

            if self.use_cv_test_transforms:
                logger.info(
                    "\tHarmonizing fold test data (use_cv_test_transforms)")
                if 'pred_harm_target' in self.stack_model_features:
                    t_cv_harm = self._nh_model.transform(
                        X_test, y_test, sites_test, covars_test)  # type: ignore
                    X_cv_harmonized[test_index, :] = t_cv_harm  # type: ignore

                if 'pred_harm_notarget' in self.stack_model_features:
                    t_cv_harm = self._nh_model.transform(
                        X_test, y_test, sites_test, covars_test)  # type: ignore
                    X_cv_harmonized_notarget[test_index, :] = t_cv_harm
                
                logger.info("\tFold test data harmonized")

        # Train the prediction model on all the harmonized data
        # Train the harmonization model on all the data
        logger.info("Fitting predictive models on all data")
        if 'pred_harm_target' in self.stack_model_features:
            logger.info("pred_harm_target")
            if self.use_cv_test_transforms:
                logger.info("Using harmonized test data (use_cv_test_transforms)")
                X_harmonized = X_cv_harmonized  # type: ignore
            else:
                logger.info("Fitting neuroHarmonize model on all data")
                X_harmonized = \
                    self._nh_model['model'].fit_transform(X, y, sites, covars)
            self.pred_model['pred_harm_target'].fit(X_harmonized, y)  # type: ignore
        
        if 'pred_harm_notarget' in self.stack_model_features:
            logger.info("pred_harm_notarget")
            if self.use_cv_test_transforms:
                X_harmonized_notarget = X_cv_harmonized_notarget
            else:
                X_harmonized_notarget = \
                    self._nh_model['notarget'].fit_transform(X, None, sites, covars)
            self.pred_model['pred_harm_notarget'].fit(X_harmonized_notarget, y)
        
        if 'pred_orig' in self.stack_model_features:
            self.pred_model['pred_orig'].fit(X, y)    
        
        logger.info("Fitting stacking model")
        # train the stack model using all stack_features
        if 'orig' in self.stack_model_features:
            cv_preds = np.hstack((cv_preds, X))
        # train the stack model that uses the predictions from CV
        self.stack_model.fit(cv_preds, y)  # type: ignore
        logger.info("Fit done")
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
        self: JuHarmonizeCV
            the fitted model
        """
        check_consistency(X, sites, y, covars, need_y=True)

        self._prepare_fit(X, y, sites, covars)
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
        check_consistency(X, sites, y, covars, need_y=True)

        self._prepare_fit(X, y, sites, covars)
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

        logger.info("Transforming data ({X.shape})")
        check_consistency(X, sites, covars=covars)
        logger.info("Predicting")
        preds = self._get_predictions(X, sites, covars)
        logger.info("Using stacked model to predict targets")
        pred_y = self.stack_model.predict(preds)  # type: ignore
        self._pred_y = pred_y
        logger.info("Harmonizing using the predicted sites")
        X_harmonized = self._nh_model.transform(X, pred_y, sites, covars)
        logger.info("Transform done")
        return X_harmonized

    def _get_predictions(self, X, sites, covars):
        """ Implementation of prediction step """
        preds = np.empty((X.shape[0], 0))
        if 'pred_harm_target' in self.stack_model_features:
            p_sites = [None]
            if self.predict_ignore_site:
                p_sites = self._sites
            
            for _, a_site in enumerate(p_sites):
                if a_site is None:
                    t_sites = sites
                else:
                    # Pretend that all samples are from a_site
                    t_sites = np.array([a_site] * len(sites))
                #s_preds = np.ones((X.shape[0], len(self._classes))) * -1
                s_preds = np.empty((X.shape[0], 0))
                for i_cls, t_cls in enumerate(self._classes):                
                    t_y = np.ones(X.shape[0], dtype=int) * t_cls
                    t_X_harmonized = self._nh_model['model'].transform(  # type: ignore
                        X, t_y, t_sites, covars)
                    #s_preds[:, i_cls] = \
                    #    self._pred_model_predict(t_X_harmonized)
                    s_preds_i = self._pred_model_predict(t_X_harmonized, model='pred_harm_target')
                    s_preds = np.column_stack((s_preds, s_preds_i))
                    
                preds = np.column_stack((preds, s_preds))
            
        if 'pred_harm_notarget' in self.stack_model_features:            
            t_X_harmonized = self._nh_model['notarget'].transform(  # type: ignore
                    X, None, sites, covars)
            s_preds = self._pred_model_predict(t_X_harmonized, model='pred_harm_notarget')
            preds = np.column_stack((preds, s_preds))
            
        if 'pred_orig' in self.stack_model_features:
            s_preds = self._pred_model_predict(X, model='pred_orig')
            preds = np.column_stack((preds, s_preds))
        
        return preds

    def _pred_model_predict(self, X):
        raise NotImplementedError("Implement in subclass")

    def predict(
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
        if self._nh_model['model'] is None:
            raise RuntimeError("Model not fitted")
        logger.info("Predicting data ({X.shape})")
        check_consistency(X, sites, covars=covars)
        logger.info("Predicting")
        preds = self._get_predictions(X, sites, covars)
        if 'pred_orig' in self.stack_model_features:
            preds = np.column_stack((preds, X))
        logger.info("Using stacked model to predict targets")
        pred_y = self.stack_model.predict(preds)
        logger.info("Predict done")
        return pred_y
