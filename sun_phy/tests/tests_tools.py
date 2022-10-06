# Testing utilities
from itertools import takewhile
import numpy as np


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, round=-1):
    """
    Compare arrays based on the following criterias :

    - Size
    - Equality
    - Percentage of non-equality and 2-norm of the error
    - Number of valid samples at the beginning
    - Number of valid samples at the end 


    Parameters
    ----------
    arr1 : ndarray
    arr2 : ndarray
    round : int
        Number of decimals to the round the values to. Disabled by default (-1)
    """

    if arr1.ndim > 1 or arr2.ndim > 1:
        raise ValueError(
            "Cannot compare arrays with dimensions bigger than one")

    # Compare the shape
    assert arr1.size == arr2.size, f"Arrays aren't the same size {arr1.size} / {arr2.size}"
    # Compare equality of the arrays
    if arr1.size == arr2.size:
        non_equality = np.sum(arr1 != arr2)
        error = np.sum(np.abs(arr1 - arr2))
        error_message = f"(Different by {error:.4f} on {non_equality} values ({non_equality / arr1.size *100:.1f} %))"
    else:
        error_message = ""

    def _round_compare(a1, a2, round):
        if round >= 0:
            return np.round(a1, round) == np.round(a2, round)
        else:
            return a1 == a2

    valid_start = True
    valid_end = True
    valid_start_count = 0
    valid_end_count = 0
    min_length = np.min([arr1.size, arr2.size])
    for i in range(min_length):
        # Compare the number of valid samples at the beginning
        if valid_start and _round_compare(arr1[i], arr2[i], round):
            valid_start_count += 1
        else:
            valid_start = False
        # Compare the number of valid samples at the end
        if valid_end and _round_compare(arr1[-(i+1)], arr2[-(i+1)], round):
            valid_end_count += 1
        else:
            valid_end = False

    assert np.array_equal(
        arr1, arr2), f"Arrays aren't equal {error_message}.\n\
            {valid_start_count} values ({valid_start_count / min_length * 100:.1f} %) of the start values are correct.\n\
                {valid_end_count} values ({valid_end_count / min_length * 100:.1f} %) of the end values are correct"
