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

def check_binary_array(bits):
    """
    Check an input binary array and returns a numpy array of uint8s

    Parameters
    ----------
    bits : ndarray or list
        List of input bits
    
    Returns
    -------
    output : ndarray
        List of bits (np.uint8)
    """
    # Checks
    if isinstance(bits, list):
        # List of bits, convert to numpy array
        bits = np.asarray(bits)
    if isinstance(bits, np.ndarray):
        # Numpy array of bits
        if bits.ndim != 1:
            raise ValueError(f"Cannot convert {bits.ndim}-dimensional data ({bits.shape})")
        elif bits.min() < 0 or bits.max() > 1:
            raise ValueError(f"Cannot convert values ranging from {bits.min()} to {bits.max()} to binary")
        bits = bits.astype(np.uint8)
    else:
        raise TypeError(f"Invalid input type : {type(bits)}")
    
    return bits




def to_binary_array(arr):
    """
    Convert a bytearray, bytes or unsiged int array/list to binary array

    Parameters
    ----------
    arr : bytearray, bytes, ndarray, list
        Input array
    binary : bool
        Input array type. True : list of bits, False : list of octets

    Returns
    -------
    output : ndarray
        Output array of bits
    """

    # Phase 1 : Convert to ndarray of x (bits or octets)
    if isinstance(arr, list):
        arr_temp = np.asarray(arr, dtype=int)
    elif isinstance(arr, bytearray) or isinstance(arr, bytes):
        arr_temp = np.asarray(list(arr), dtype=int)
    elif isinstance(arr, np.ndarray):
        arr_temp = arr.astype(int)
    elif isinstance(arr, str):
        raise TypeError("Cannot use string type, please use a bytearray")
    else:
        raise TypeError(f"Invalid input array type : {type(arr)}")

    # Phase 2 : convert to binary
    bit_list = []
    if arr_temp.min() < 0 or arr_temp.max() > 255:
        raise ValueError(
            "Invalid byte_message range. It should be between 0 and 255")
    
    for byte in arr_temp:
        # The order of the bits must be reversed
        bit_list += [int(x) for x in np.binary_repr(byte, 8)[::-1]]
    
    output = np.array(bit_list)

    return output
