import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Union, List


def subset_data(
    index: Union[npt.NDArray, List[int]],
    X: npt.NDArray,
    sites: npt.NDArray,
    y: Optional[npt.NDArray] = None,
    covars: Optional[npt.NDArray] = None,
) -> Tuple[
    npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]
]:
    assert not isinstance(index, int)
    _X = X[index]
    _sites = sites[index]
    _y = None
    _covars = None
    if y is not None:
        _y = y[index]
    if covars is not None:
        _covars = covars[index]
    return _X, _sites, _y, _covars


def check_consistency(
    X: npt.NDArray,
    sites: npt.NDArray,
    y: Optional[npt.NDArray] = None,
    covars: Optional[npt.NDArray] = None,
    need_y: bool = False,
) -> None:
    """Check that the dimensions of the data are consistent."""
    assert X.shape[0] == sites.shape[0], (
        f"X and sites must have the same number of samples: {X.shape[0]}, "
        f"{sites.shape[0]}"
    )

    assert need_y is False or y is not None, "y must be provided"
    if y is not None:
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same number of samples"

    if covars is not None:
        assert (
            covars.shape[0] == X.shape[0]
        ), "covars and X must have the same number of samples"


def check_harmonization_results(
    X: npt.NDArray,
    harmonized_X: npt.NDArray,
    sites: npt.NDArray,
    y: Optional[npt.NDArray] = None,
) -> None:
    if np.isnan(harmonized_X).any() or np.isinf(harmonized_X).any():
        print("Warning: NaNs or Infs in harmonized data")
        print(f"Sites: {np.unique(sites)}")
        if y is not None:
            print(f"Targets: {np.unique(y)}")
        data_colvar = np.var(X, axis=0)
        print(
            f"Data columns with low variance: {np.sum(data_colvar < 1e-6)}",
        )
        raise RuntimeError("Harmonization of trainig data failed in CV!")
