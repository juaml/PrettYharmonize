import numpy as np
from seaborn import load_dataset

from prettyharmonize import PrettYharmonizePredictor


def test_PrettYharmonizePredictor() -> None:
    """Test PrettYharmonizePredictor Class"""
    # Load data
    df_iris = load_dataset("iris")

    # Binary clasification problem
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

    # Get target
    target = df_iris["species"].isin(["versicolor"]).astype(int)
    target = target.to_numpy()
    # Get Classes
    # classes = np.unique(target)
    # Get number of classe
    # n_classes = len(classes)

    # Data must be a numpy array [N_samples x N_Features]
    data = df_iris[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].values

    # Get samples and features form the data
    num_samples, _ = data.shape

    # add some systematic noise to the data
    # data_harm = data + np.random.normal(loc=1, scale=0.2, size=data.shape)
    sites = np.random.randint(low=0, high=2, size=num_samples)

    # test Harmonization implementation
    pretty_model = PrettYharmonizePredictor()
    assert pretty_model._models is None

    pretty_model.fit(data, sites=sites, y=target)

    assert len(pretty_model._models) == data.shape[1]

    harm_data_ft = pretty_model.transform(data)
    assert harm_data_ft.shape == data.shape

    # test Harmonization implementation with extra vars
    pretty_model = PrettYharmonizePredictor()
    assert pretty_model._models is None

    pretty_model.fit(data, sites=sites, y=target)

    assert len(pretty_model._models) == data.shape[1]

    harm_data_ft = pretty_model.transform(data)
    assert harm_data_ft.shape == data.shape


def test_PrettYharmonizePredictor_storage() -> None:
    """Test PrettYharmonizePredictor Class using storage."""
    # Load data
    df_iris = load_dataset("iris")

    # Binary clasification problem
    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

    # Get target
    target = df_iris["species"].isin(["versicolor"]).astype(int)
    target = target.to_numpy()
    # Get Classes
    # classes = np.unique(target)
    # Get number of classe
    # n_classes = len(classes)

    # Data must be a numpy array [N_samples x N_Features]
    data = df_iris[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].values

    # Get samples and features form the data
    num_samples, _ = data.shape

    # add some systematic noise to the data
    # data_harm = data + np.random.normal(loc=1, scale=0.2, size=data.shape)
    sites = np.random.randint(low=0, high=2, size=num_samples)

    # test Harmonization implementation
    pretty_model = PrettYharmonizePredictor(use_disk=True)
    assert pretty_model._models is None

    pretty_model.fit(data, sites=sites, y=target)

    assert len(pretty_model._models) == data.shape[1]

    harm_data_ft = pretty_model.transform(data)
    assert harm_data_ft.shape == data.shape

    # test Harmonization implementation with extra vars
    pretty_model = PrettYharmonizePredictor(use_disk=True)
    assert pretty_model._models is None

    pretty_model.fit(data, sites=sites, y=target)

    assert len(pretty_model._models) == data.shape[1]

    harm_data_ft = pretty_model.transform(data)
    assert harm_data_ft.shape == data.shape
