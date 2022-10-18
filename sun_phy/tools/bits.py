import numpy as np

def to_bin(number, width=8, MSB_first=True):
    """
    Converts a number to binary representation

    Parameters
    ----------
    number : int

    Returns
    -------
    output : ndarray
        Array of bits
    """

    return np.array([int(x) for x in np.binary_repr(number, width)[::(1 if MSB_first else -1)]])

def to_int(array, width=8, MSB_first=True):
    """
    Convert a binary array to integer

    Parameters
    ----------
    array : ndarray
        Array of bits

    Returns
    -------
    number : int
    """
    return np.packbits(array[::(1 if MSB_first else -1)], bitorder='little')[0]


def from_bitstring(bit_string: str) -> np.ndarray:
    """
    Converts a bitstring to bit array

    Spaces are removed
    """

    return np.array([int(x) for x in bit_string.replace(' ', '')], dtype=np.uint8)