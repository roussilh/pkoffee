import pytest 
import numpy as np

def test_size_mismatch_valid():
    from pkoffee.metrics import check_size_match

    a: np.ndarray = np.array([1, 2, 3])
    b: np.ndarray = np.array([2, 3, 4]) 

    check_size_match(array_a = a, array_b = b)


def test_size_mismatch_invalid():
    from pkoffee.metrics import check_size_match, SizeMismatchError

    a: np.ndarray = np.zeros(shape=5)
    b: np.ndarray = np.ones(shape = 10)
    with pytest.raises(SizeMismatchError, match="Arrays must have same length"):
        check_size_match(a, b)
  


def test_r2():
    from pkoffee.metrics import compute_r2

    rng = np.random.default_rng(seed=0)
    y_true: np.ndarray = rng.normal(size=10)
    assert compute_r2(y_true, y_pred =y_true) ==1

    y_pred: np.ndarray = np.zeros_like(y_true)
    assert compute_r2(y_true, y_pred) < 0.0

def test_rmse():
    from pkoffee.metrics import compute_rmse

    rng = np.random.default_rng(seed=0)
    y_true: np.ndarray = rng.normal(size = 10)
    assert compute_rmse(y_true, y_pred=y_true) == 0

    y_pred: np.ndarray = np.zeros_like(y_true)
    assert compute_rmse(y_true, y_pred) > 0

def test_mae():
    from pkoffee.metrics import compute_mae

    rng = np.random.default_rng(seed=0)
    y_true: np.ndarray = rng.normal(size=10)
    assert compute_mae(y_true, y_pred =y_true) ==0

    y_pred: np.ndarray = np.zeros_like(y_true)
    assert compute_mae(y_true, y_pred) > 0