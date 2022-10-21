import numpy as np
from numpy import poly1d, polymul, polydiv, polyadd



RATE_BITS = slice(0, 4+1)
FRAME_LENGTH_BITS = slice(6, 16+1)
SCRAMBLER_BITS = slice(19, 20+1)
HCS_BITS = slice(22, 29+1)
TAIL_BITS = slice(30, 35+1)


def _uintarr(value, bits=8, order='big'):
    """
    Returns an array corresponding to the binary representation of a given unsigned integer

    Parameters
    ----------
    value : int
        Value to convert
    bits : int
        number of bits in the output
    order : str
        'big' means the LSB is last -> [-1] and the array is "readable" normally
        'little' means the LSB is first -> [0] and the array is flipped
    """
    # Creating a string representation of the number
    string = np.binary_repr(value)
    if len(string) > bits:
        raise ValueError(
            f"Number is too big for the given number of bits : {bits}")
    elif len(string) < bits:
        # Adding padding zero
        pad_string = '0'*(bits-len(string)) + string
    else:
        pad_string = string
    # Convert to numpy array
    array = np.array([int(x) for x in pad_string], dtype=np.uint8)

    return (array if order == 'big' else array[::-1])

def HCS_calculation(input_array):
    """
    Applies HCS crc to input array
    See 18.2.1.3 PHR

    Parameters
    ----------
    input_array : ndarray
        Input values (0 and 1) in a numpy array. [0] is the first element
    
    Returns
    -------
    output_array : ndarray
        Calculated CRC
    """
    # Generator polynomial x^8 + x^2 + x + 1
    G = poly1d([1, 0, 0, 0, 0, 0, 1, 1, 1])
    # part a)
    k = poly1d(np.block([1, np.zeros(22)]))
    b = poly1d(np.ones(8))
    c = polymul(k, b)
    _, A = polydiv(c, G)
    A = np.mod(A, 2).astype(int)
    # part b)
    a = polymul(poly1d(input_array), poly1d(np.block([1, np.zeros(8)])))
    _, B = polydiv(a, G)
    B = np.mod(B, 2).astype(int)

    # one's complement of the modulo-2 sum    
    output_array = 1 - np.mod(polyadd(A, B), 2)
    return output_array


class PHR():
    def __init__(self, rate, length, scrambler, phr_length):
        """
        PHY Header for MR-OFDM
        See 18.2.1.3 in 802.15.4g specification

        Parameters
        ----------
        rate : int
            0-4 (RA4-RA0) Specifies the data rate of the payload (=MCS)
        length : int
            6-16 (L10-L0) Frame length. Total number of octets contained in the PSDU (prior to FEC encoding)
        scrambler : int
            19-20 (S1-S0) Scrambler  Scrambling seed (0 - 3)
        phr_length : int
            Length of the PHR (to match the number of symbols wanted)
        """
        # Check values and types
        if not isinstance(rate, int):
            raise ValueError("Invalid rate type")
        if not (0 <= rate <= 6):
            raise ValueError(f"Invalid rate value {rate}")
        if not isinstance(length, int):
            raise ValueError("Invalid length type")
        if not (1 <= length <= 2**11):
            raise ValueError(f"Invalid length value {length}")
        if not isinstance(scrambler, int):
            raise ValueError("Invalid scrambler type")
        if not (0 <= scrambler <= 3):
            raise ValueError(f"Invalid scrambler value {scrambler}")

        # Store values
        self._phr_length = phr_length
        self._RA = rate
        self._L = length
        self._S = scrambler

    def value(self):
        """
        Returns the byte-array value of the PHY Header with tail and pad bits

        Returns
        -------
        output : list of bytes
        """
        # PHR : 36
        # the rest is padded with 0s
        output = np.zeros([self._phr_length], dtype=np.uint8)
        # Build the array in correct order (LSB first)
        # This is great because it is the order in which it will be sent
        # as well as the order in which it's "logical" to build it (indices)

        # TODO : Check if RA is the value of MCS (MCSx => x)
        # or if it is the identifier (MCS1, Option3 => 0)
        # See table 68j
        output[RATE_BITS] = _uintarr(self._RA, bits=5)
        output[FRAME_LENGTH_BITS] = _uintarr(self._L, bits=11)
        output[SCRAMBLER_BITS] = _uintarr(self._S, bits=2)
        
        HCS = HCS_calculation(output[0:22])

        output[HCS_BITS] = HCS
        # Tail bits (30-35) are all zeros
        return output
