# %%
# Imports
import numpy as np
from numpy.testing import assert_array_equal
from seaborn import load_dataset

from prettyharmonize import PrettYharmonize


def test_PrettYharmonize() -> None:
    """Test PrettYharmonize Class"""
    # Load data
    df_iris = load_dataset('iris')

    # Binary clasification problem
    df_iris = df_iris[df_iris['species'].isin(['versicolor', 'virginica'])]

    # Get target
    target = df_iris['species'].isin(['versicolor']).astype(int)
    target = target.to_numpy()
    # Get Classes
    # classes = np.unique(target)
    # Get number of classe
    # n_classes = len(classes)

    # Data must be a numpy array [N_samples x N_Features]
    data = df_iris[['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']].values

    # Get samples and features form the data
    num_samples, _ = data.shape

    # generate random sites
    sites = np.random.randint(low=0, high=2, size=num_samples)

    # test Harmonization implementation
    prettyharm_model = PrettYharmonize()
    assert prettyharm_model._nh_model is None
    harm_data_ft = prettyharm_model.fit_transform(data, target, sites)
    assert prettyharm_model._nh_model is not None
    assert prettyharm_model._need_covars is False
    harm_data_t = prettyharm_model.transform(data, target, sites)
    assert_array_equal(harm_data_ft, harm_data_t)

    # test using string in sites
    sites = np.random.choice(['site1', 'site2'], size=num_samples)

    # test Harmonization implementation
    prettyharm_model = PrettYharmonize()
    assert prettyharm_model._nh_model is None
    harm_data_ft = prettyharm_model.fit_transform(data, target, sites)
    assert prettyharm_model._nh_model is not None
    assert prettyharm_model._need_covars is False
    harm_data_t = prettyharm_model.transform(data, target, sites)
    assert_array_equal(harm_data_ft, harm_data_t)
