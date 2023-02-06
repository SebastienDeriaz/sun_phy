# MR-O-QPSK Modulator
#
# Sébastien Deriaz
# 12.10.2022
#
# this file layout :
#
# - tool functions
# - 802.15.4g constants
# - mr-o-qpsk modulator class

from email import message
import numpy as np
from enum import Enum

from ..tools.errors import UnsupportedError
from ..tools.bits import to_bin, to_int, from_bitstring, check_binary_array, to_binary_array
from ..tools.modulations import BPSK

from colorama import Fore

class Frequency_band(Enum):
    Band_470MHz = 5,
    Band_780MHz = 1,
    Band_868MHz = 6,
    Band_915MHz = 2,
    Band_920MHz = 7,
    Band_917MHz = 3,
    Band_950MHz = 8
    Band_2450MHz = 4


# Length of the preamble in bits as a function of the frequency band
# See 18.3.1.1
PREAMPLE_LENGTH = {
    Frequency_band.Band_470MHz: 32,  # 470-510
    Frequency_band.Band_780MHz: 56,  # 779-787
    Frequency_band.Band_868MHz: 32,  # 868-870
    Frequency_band.Band_915MHz: 56,  # 902-928
    Frequency_band.Band_917MHz: 56,  # 917-923.5
    Frequency_band.Band_920MHz: 32,  # 920-928
    Frequency_band.Band_950MHz: 32,  # 950-958
    Frequency_band.Band_2450MHz: 56  # 2400-2438.5
}

# See 18.3.1.2
SFD = from_bitstring('1110101101100010')

PHR_LENGTH = 3

# Chip values as a function of input value and DSSS type
# (N,M) means there's only one possibility for this size combination
# (N,M)x means there's multiple possibilities
DSSS = {
    '(2,1)': {
        '0': '10',
        '1': '01'
    },
    '(4,1)': {
        '0': '1010',
        '1': '0101'
    },
    '(8,1)0': {
        '0': '10110001',
        '1': '01001110'
    },
    '(8,1)1': {
        '0': '01100011',
        '1': '10011100'
    },
    '(16,1)0': {
        '0': '0010001111010110',
        '1': '1101110000101001',
    },
    '(16,1)1': {
        '0': '0100011110101100',
        '1': '1011100001010011'
    },
    '(32,1)0': {
        '0': '11011110101000100111000001100101',
        '1': '00100001010111011000111110011010'
    },
    '(32,1)1': {
        '0': '11101111010100010011100000110010',
        '1': '00010000101011101100011111001101'
    },
    '(64,1)': {
        '0': '1011001000100101101100011101000011010111001111011111000000101010',
        '1': '0100110111011010010011100010111100101000110000100000111111010101'
    },
    '(128,1)': {
        '0': '10011000100010110100111001000010010100100110110111000111101000001101010001100101110110000111010111100111110111111000000010101011',
        '1': '01100111011101001011000110111101101011011001001000111000010111110010101110011010001001111000101000011000001000000111111101010100'
    },
    '(8,4)': {
        '0000': '00000001',
        '1000': '11010000',
        '0100': '01101000',
        '1100': '10111001',
        '0010': '11100101',
        '1010': '00110100',
        '0110': '10001100',
        '1110': '01011101',
        '0001': '10100010',
        '1001': '01110011',
        '0101': '11001011',
        '1101': '00011010',
        '0011': '01000110',
        '1011': '10010111',
        '0111': '00101111',
        '1111': '11111110'
    },
    '(16,4)': {
        '0000': '0011111000100101',
        '1000': '0100111110001001',
        '0100': '0101001111100010',
        '1100': '1001010011111000',
        '0010': '0010010100111110',
        '1010': '1000100101001111',
        '0110': '1110001001010011',
        '1110': '1111100010010100',
        '0001': '0110101101110000',
        '1001': '0001101011011100',
        '0101': '0000011010110111',
        '1101': '1100000110101101',
        '0011': '0111000001101011',
        '1011': '1101110000011010',
        '0111': '1011011100000110',
        '1111': '1010110111000001'
    },
    '(32,4)': {
        '0000': '11011001110000110101001000101110',
        '1000': '11101101100111000011010100100010',
        '0100': '00101110110110011100001101010010',
        '1100': '00100010111011011001110000110101',
        '0010': '01010010001011101101100111000011',
        '1010': '00110101001000101110110110011100',
        '0110': '11000011010100100010111011011001',
        '1110': '10011100001101010010001011101101',
        '0001': '10001100100101100000011101111011',
        '1001': '10111000110010010110000001110111',
        '0101': '01111011100011001001011000000111',
        '1101': '01110111101110001100100101100000',
        '0011': '00000111011110111000110010010110',
        '1011': '01100000011101111011100011001001',
        '0111': '10010110000001110111101110001100',
        '1111': '11001001011000000111011110111000'
    }

}

# Covering code as a function of n (MDSSS)
COVERING_CODE = {
    4: DSSS['(16,1)0']['0'],
    8: DSSS['(32,1)0']['0'],
    16: DSSS['(64,1)']['0'],
    32: DSSS['(128,1)']['0']
}


def _hadamard_matrix(N):
    """
    Return hadamard matrix of size N

    Parameters
    ----------
    N : int
        Size of the hadamard matrix (2**x)

    Returns
    -------
    M : ndarray
        Hadamard matrix
    """

    iterations = int(np.log2(N))
    H = np.array([1])
    for _ in range(iterations):
        H = np.block([[H, H], [H, -H]])

    return H


def _hadamard_codeword(N, M, i):
    """
    Return the ith codeword of a [N, M] Hadamard encoder

    Parameters
    ----------
    N : int
        Size of the hadamard matrix
    M : int
        Number of bits used as row indices in the hadamard matrix
    i : int / list of int
        Index / indices of the hadamard codeword

    Returns
    -------
    codeword : ndarray / list of ndarray
    """
    H = _hadamard_matrix(N)

    def cw(H, M, i):
        N_indices = 2**M
        if i < N_indices // 2:
            # Return a row of the matrix
            return H[i, :]
        else:
            # Return an inverted row the matrix (with inverted index)
            iinv = N_indices - 1 - i
            return np.logical_not(H[iinv, :])

    if isinstance(i, int):
        # Single codeword
        codeword = cw(H, M, i)
    else:
        # list of codewords
        codeword = [cw(H, M, ii) for ii in i]

    return codeword


PILOT_SEQUENCES = {
    Frequency_band.Band_470MHz: from_bitstring('1101 1110 1010 0010 0111 0000 0110 0101'),
    Frequency_band.Band_780MHz: from_bitstring('1011 0010 0010 0101 1011 0001 1101 0000 1101 0111 0011 1101 1111 0000 0010 1010'),
    Frequency_band.Band_868MHz: from_bitstring('1101 1110 1010 0010 0111 0000 0110 0101'),
    Frequency_band.Band_915MHz: from_bitstring('1011 0010 0010 0101 1011 0001 1101 0000 1101 0111 0011 1101 1111 0000 0010 1010'),
    Frequency_band.Band_917MHz: from_bitstring('1011 0010 0010 0101 1011 0001 1101 0000 1101 0111 0011 1101 1111 0000 0010 1010'),
    Frequency_band.Band_920MHz: from_bitstring('1101 1110 1010 0010 0111 0000 0110 0101'),
    Frequency_band.Band_950MHz: from_bitstring('1101 1110 1010 0010 0111 0000 0110 0101'),
    Frequency_band.Band_2450MHz: from_bitstring(
        '1001 1000 1000 1011 0100 1110 0100 0010 0101 0010 0110 1101 1100 0111 1010 0000 1101 0100 0110 0111 1101 1000 0111 0101 1110 0111 1101 1111 1000 0000 1010 1011')
}

# MP
PILOT_SPACING = {
    Frequency_band.Band_470MHz: 512,
    Frequency_band.Band_780MHz: 1024,
    Frequency_band.Band_868MHz: 512,
    Frequency_band.Band_915MHz: 1024,
    Frequency_band.Band_917MHz: 1024,
    Frequency_band.Band_920MHz: 512,
    Frequency_band.Band_950MHz: 512,
    Frequency_band.Band_2450MHz: 2048
}


class SpreadingMode(Enum):
    DSSS = 0
    MDSSS = 1


# Spreading when DSSS is used as a function of frequency band
# A tuple means the codes should be used alternatively
SHR_SPREADING = {
    Frequency_band.Band_470MHz: ('(32,1)0'),
    Frequency_band.Band_780MHz: ('(64,1)'),
    Frequency_band.Band_868MHz: ('(32,1)0'),
    Frequency_band.Band_915MHz: ('(64,1)'),
    Frequency_band.Band_917MHz: ('(64,1)'),
    Frequency_band.Band_920MHz: ('(32,1)0', '(32,1)1'),
    Frequency_band.Band_950MHz: ('(32,1)0'),
    Frequency_band.Band_2450MHz: ('(128,1)')
}

# For PHR and PSDU, use ()0 and ()1 alternatively (tuple)
PHR_SPREADING = {
    Frequency_band.Band_470MHz: ('(8,1)0', '(8,1)1'),
    Frequency_band.Band_780MHz: ('(16,1)0', '(16,1)1'),
    Frequency_band.Band_868MHz: ('(8,1)0', '(8,1)1'),
    Frequency_band.Band_915MHz: ('(16,1)0', '(16,1)1'),
    Frequency_band.Band_917MHz: ('(16,1)0', '(16,1)1'),
    Frequency_band.Band_920MHz: ('(8,1)0', '(8,1)1'),
    Frequency_band.Band_950MHz: ('(8,1)0', '(8,1)1'),
    Frequency_band.Band_2450MHz: ('(16,1)0', '(16,1)1')
}

# DSSS spreading for PSDU as a function of frequency band and RateMode
PSDU_SPREADING_DSSS = {
    Frequency_band.Band_470MHz: {
        0: ('(8,1)0', '(8,1)1'),
        1: ('(4,1)'),
        2: ('(2,1)'),
        3: False
    },
    Frequency_band.Band_780MHz: {
        0: ('(16,1)0', '(16,1)1'),
        1: ('(16,4)'),
        2: ('(8,4)'),
        3: False
    },
    Frequency_band.Band_868MHz: {
        0: ('(8,1)0', '(8,1)1'),
        1: ('(4,1)'),
        2: ('(2,1)'),
        3: False
    },
    Frequency_band.Band_915MHz: {
        0: ('(16,1)0', '(16,1)1'),
        1: ('(16,4)'),
        2: ('(8,4)'),
        3: False
    },
    Frequency_band.Band_917MHz: {
        0: ('(16,1)0', '(16,1)1'),
        1: ('(16,4)'),
        2: ('(8,4)'),
        3: False
    },
    Frequency_band.Band_920MHz: {
        0: ('(8,1)0', '(8,1)1'),
        1: ('(4,1)'),
        2: ('(2,1)'),
        3: False
    },
    Frequency_band.Band_950MHz: {
        0: ('(8,1)0', '(8,1)1'),
        1: ('(4,1)'),
        2: ('(2,1)'),
        3: False
    },
    Frequency_band.Band_2450MHz: {
        0: ('(32,1)0', '(32,1)1'),
        1: ('(32,4)'),
        2: ('(16,4)'),
        3: ('(8,4)')
    },
}
# MDSSS Spreading for PSDU as a function of frequency band a rate mode
# Since it's always (N, 8), only N is stored
PSDU_SPREADING_FACTOR_MDSSS = {
    Frequency_band.Band_470MHz: "not supported",
    Frequency_band.Band_780MHz: {
        0: 64,
        1: 32,
        2: 32,
        3: 16
    },
    Frequency_band.Band_868MHz: "not supported",
    Frequency_band.Band_915MHz: {
        0: 64,
        1: 32,
        2: 32,
        3: 16
        
    },
    Frequency_band.Band_917MHz: {
        0: 64,
        1: 32,
        2: 32,
        3: 16
    },
    Frequency_band.Band_920MHz: "not supported",
    Frequency_band.Band_950MHz: "not supported",
    Frequency_band.Band_2450MHz: {
        0: 128,
        1: 64,
        2: 64,
        3: 32
    },
}

# BDE Use as a function of FrequencyBand and RateMode
# BDE is always False if MDSSS is used
PSDU_BDE_DSSS = {
    Frequency_band.Band_470MHz: {
        0: True,
        1: True,
        2: True,
        3: False
    },
    Frequency_band.Band_780MHz: {
        0: True,
        1: False,
        2: False,
        3: False,
    },
    Frequency_band.Band_868MHz: {
        0: True,
        1: True,
        2: True,
        3: False,
    },
    Frequency_band.Band_915MHz: {
        0: True,
        1: False,
        2: False,
        3: False,
    },
    Frequency_band.Band_917MHz: {
        0: True,
        1: False,
        2: False,
        3: False,
    },
    Frequency_band.Band_920MHz: {
        0: True,
        1: True,
        2: True,
        3: False,
    },
    Frequency_band.Band_950MHz: {
        0: True,
        1: True,
        2: True,
        3: False,
    },
    Frequency_band.Band_2450MHz: {
        0: True,
        1: False,
        2: False,
        3: False,
    }
}

# Data rate as a function of spreading mode, frequency band and rate mode
# bit / s
DATA_RATE = {
    SpreadingMode.DSSS.value : {
        Frequency_band.Band_470MHz : {
            0 : 6250,
            1 : 12500,
            2 : 25000,
            3 : 50000
        },
        Frequency_band.Band_780MHz : {
            0 : 31250,
            1 : 125000,
            2 : 250000,
            3 : 500000
        },
        Frequency_band.Band_868MHz : {
            0 : 6250,
            1 : 12500,
            2 : 25000,
            3 : 50000
        },
        Frequency_band.Band_915MHz : {
            0 : 31250,
            1 : 125000,
            2 : 250000,
            3 : 500000
        },
        Frequency_band.Band_917MHz : {
            0 : 31250,
            1 : 125000,
            2 : 250000,
            3 : 500000
        },
        Frequency_band.Band_920MHz : {
            0 : 6250,
            1 : 12500,
            2 : 25000,
            3 : 50000
        },
        Frequency_band.Band_950MHz : {
            0 : 6250,
            1 : 12500,
            2 : 25000,
            3 : 50000
        },
        Frequency_band.Band_2450MHz : {
            0 : 31250,
            1 : 125000,
            2 : 250000,
            3 : 500000
        }
    },
    SpreadingMode.MDSSS.value : {
        Frequency_band.Band_470MHz : "not supported",
        Frequency_band.Band_780MHz : {
            0 : 62500,
            1 : 125000,
            2 : 250000,
            3 : 500000
        },
        Frequency_band.Band_868MHz : "not supported",
        Frequency_band.Band_915MHz : {
            0 : 62500,
            1 : 125000,
            2 : 250000,
            3 : 500000
        },
        Frequency_band.Band_917MHz : {
            0 : 62500,
            1 : 125000,
            2 : 250000,
            3 : 500000
        },
        Frequency_band.Band_920MHz : "not supported",
        Frequency_band.Band_950MHz : "not supported",
        Frequency_band.Band_2450MHz : {
            0 : 62500,
            1 : 125000,
            2 : 250000,
            3 : 500000
        }
    }
}

def half_sine(t, Tc):
    # 0 <= t <= 2*Tc
    f1 = lambda x : np.sin(np.pi * x / (2 * Tc))
    # otherwise
    f2 = 0

    return np.piecewise(t, np.logical_and(0 <= t, t <= 2*Tc), [f1, f2])

def raised_cosine(t, Tc):
    r = 0.8
    # evaluate the whole function, then correct for NaN values
    output = np.ones_like(t)
    mask = (t != 0)
    t_diff0 = t[mask]
    output[mask] = np.sin(np.pi * t_diff0 / Tc) / (np.pi * t_diff0 / Tc) * np.cos(r * np.pi * t_diff0 / Tc) / (1 - 4*r**2*t_diff0**2 / Tc**2) 

    return output

# Possible combinations of parameters
# spreading_mode, frequency_band
MDSSS_SUPPORTED_FREQUENCY_BANDS = {
    Frequency_band.Band_470MHz: False,
    Frequency_band.Band_780MHz: True,
    Frequency_band.Band_868MHz: False,
    Frequency_band.Band_915MHz: True,
    Frequency_band.Band_917MHz: True,
    Frequency_band.Band_920MHz: False,
    Frequency_band.Band_950MHz: False,
    Frequency_band.Band_2450MHz: True
}

class Mr_o_qpsk_modulator:
    def __init__(self, frequency_band: Frequency_band, rate_mode: int, spreading_mode: int, samples_per_symbol=10, verbose=False):
        """
        Creates an instance of a MR-FSK modulator

        Parameters
        ----------
        frequency_band : Frequency_band
        rate_mode : int
        spreading_mode : int
            Use DSSS spreading mode (0) or MDSSS (1)
        samples_per_symbol : Number of sampled value for each symbol
        verbose : bool
            Enable the printing of additionnal information
        """
        # Checks
        if spreading_mode == SpreadingMode.MDSSS.value:
            # Check if the frequency band is supported
            if not MDSSS_SUPPORTED_FREQUENCY_BANDS[frequency_band]:
                raise UnsupportedError(f"Unsupported frequency band {frequency_band} for MDSSS spreading")


        self._samples_per_symbol = samples_per_symbol
        self._SM = spreading_mode
        self._frequency_band = frequency_band
        self._RM = rate_mode
        self._verbose = verbose

    def _BDE(self, message, Eprevious, M=0):
        """
        Apply Bit Differential Envoding (BDE) to the given binary message

        Parameters
        ----------
        message : ndarray
            Binary message
        Eprevious : int (0 or 1)
            Initial state of the BDE
        M : int
            =0 if differential encoding is disabled (default)
            >0 if differential encoding is enabled. M is then the modulo value at which the encoder is disabled

        Returns
        -------
        signal : ndarray
            output
        """

        signal = np.zeros_like(message)

        for i, m in enumerate(message):
            if M > 0 and np.mod(i, M) == 0:
                # Differential encoding mode and (n mod M) = 0
                signal[i] = m
            elif i == 0:
                signal[i] = m ^ Eprevious
            else:
                signal[i] = m ^ signal[i-1]

        return signal

    def _DSSS(self, message: np.ndarray, spreading : tuple):
        """
        Apply DSSS mapping to the given binary signal

        Parameters
        ----------
        message : ndarray
            Message bitstream
        spreading : tuple
            Spreading key, for exemple ('(16,1)0', '(16,1)1') or ('(8,1)')
            Each value of the tuple is used atlernatively
        
        Returns
        -------
        output : ndarray
            Output bitstream
        """

        # If spreading is not a tuple of spreading modes, make it so
        if not isinstance(spreading, tuple):
            spreading = (spreading,)

        # Number of spreading mode (one or two)
        N_spreads = len(spreading)

        def arr_to_bitarray(arr):
            """
            Convert a numpy array to a bit array
            example : [0, 0, 1, 0, 1] -> '00101'
            """
            return ''.join([str(x) for x in arr])

        # key length of the first spreading dictionary
        N = len(list(DSSS[spreading[0]].keys())[0])
        # output length of the first spreading dictionary
        x = len(list(DSSS[spreading[0]].values())[0])

        # Take each N-sized block and convert it with the DSSS dictionary
        # Then convert to integer array and concatenate
        # ['01', '00', '10', ...] -> ['010010101001...', '0100011100...', '0100110101001', ...] ->
        # [[0,1,0,0,1,1,0,1,...], [0,1,0,0,0,1,1,1,0,0,...], [0,1,0,0,1,1,0,1,0,1,0,0,1,...],...] ->
        # [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, ...]

        output = []
        # Loop over all inputs bits (size N)
        for i, input_bits in enumerate(message.reshape(-1, N)):
            spread_index = i % N_spreads
            output += DSSS[spreading[spread_index]
                           ][arr_to_bitarray(input_bits)]

        return np.array(output).astype(int)

    def _hadamard_encoder(self, bitstreams, N):
        """
        Converts input bitstreams into output bitstreams

        b⁰0 b⁰1 b⁰2     t⁰0 t⁰1 ...
        b¹0 b¹1 b¹2  -> t¹0 t¹1 ... 
        b²0 b²1 b²2     t²0 t²1 ...

        Parameters
        ----------
        bitstreams : ndarray
            Matrix of bitstreams (3 rows)

        Returns
        -------
        output : ndarray
            Matrix of output bitstreams (3 rows)
        """
        output = np.zeros([3, bitstreams.shape[1] // 3 * N], dtype=int)
        for ib, bs in enumerate(bitstreams):
            # Convert bitstreams to a list of indices
            indices = [to_int(x) for x in bs.reshape(-1, 3, order='A')]
            codewords_list = _hadamard_codeword(N, 3, indices)
            output[ib, :] = np.concatenate(codewords_list)

        return output

    def _SPC_encoder(self, bitstreams):
        """
        Applies SPC Encoder to the input bitstreams

        t⁰0 t⁰1 ...      t⁰0 t⁰1 ...
        t¹0 t¹1 ...   -> t¹0 t¹1 ... 
        t²0 t²1 ...      t²0 t²1 ...
                         t³0 t³1 ...

        Parameters
        ----------
        bitstreams : ndarray
            Input bitstreams (3 rows)
        
        Returns
        -------
        output : ndarray
            Output bitstreams (4 rows)
        """
        # TUnit is applied on each 3x4 block of data
        output = np.zeros([4, bitstreams.shape[1]], dtype=np.uint8)

        for i in range(bitstreams.shape[1] // 4):
            block_in = bitstreams[:, i*4:(i+1)*4]
            # Copy the upper 3x4 bits
            output[:3, i*4:(i+1)*4] = block_in
            # Create the new parity row
            for ic in range(4):
                output[3, i*4 + ic] = not (block_in[0, 1]
                                           ^ block_in[1, 1] ^ block_in[2, 1])

        return output

    def _TPC(self, message_binary, N):
        """
        Apply Turbo Product Code (TPC) to the binary message

                        t⁰0 t⁰1 ...
        b0 b1 b2 ... -> t¹0 t¹1 ...
                        t²0 t²1 ...
                        t³0 t³1 ...
        
        Parameters
        ----------
        message_binary : ndarray
            Input bitstream
        N : int
            Size of the hadamard encoder
        
        Returns
        -------
        output : ndarray
            Output bitstreams ()    
        """
        # NOTE : custom padding
        # If the message isn't N * 8 bits, pad it accordingly
        # This is a custom solution (not in the 802.15.4g norm) to correct the problem that arises when
        # the PSDU is passed through the interleaver and must be multiples of N_INTRLV (7*18)
        # Thus it isn't always a multiple of 8 bits and the reshape cannot happen properly
        pad = np.mod(message_binary.size, 8)
        if pad > 0:
            message_binary = np.concatenate([message_binary, np.zeros(pad, dtype=np.uint8)])
            self._print_verbose(f"Padding binary message with {pad} zeros ({message_binary.size-pad} -> {message_binary.size})")


        # Insert the p value (=0) to transform each octet into a 9-bit word
        p = 0
        message_p = np.insert(message_binary, (np.arange(message_binary.size // 8)+1) * 8, p) 

        # Split the message into corresponding bitstreams (one per row)
        message_split = message_p.reshape(3, -1, order='F')
        print(f"Splitting message {message_p.shape} -> {message_split.shape}")

        # Call Hadamard encoder
        bitstreams_3 = self._hadamard_encoder(message_split, N)
        print(f"Applying Hadamard encoder {message_split.shape} -> {bitstreams_3.shape}")

        output = self._SPC_encoder(bitstreams_3)
        print(f"Applying SPC encodee {bitstreams_3.shape} -> {output.shape}")

        return output



    def _spreading_multiplexing(self, bitstreams: np.ndarray, n):
        """
        Apply spreading and multiplexing to the input bitstreams

        Parameters
        ----------
        bitstreams : ndarray
            Input bitstreams (4 rows)
        
        Returns
        -------
        output : ndarray
            Output bitstream
        """
        ci = np.zeros(bitstreams.size, dtype=bitstreams.dtype)

        c = np.zeros_like(ci)

        h = _hadamard_matrix(4)

        for i in range(bitstreams.shape[1] // 4):
            s = np.logical_xor(bitstreams[:, i*4:(i+1)*4], h)

            ci[i*4:(i+1)*4] = np.logical_or(np.logical_and(s[0, :],
                                                           s[1, :]), np.logical_and(s[2, :], s[3, :]))

        mi = np.array(list(COVERING_CODE[n])).astype(int)

        i = np.arange(4*n, dtype=int)

        output = np.logical_xor(c[(np.mod(i, 4) + np.floor(i / n) * 4).astype(np.uint8)], mi)

        return output

    def _MDSSS(self, message, spreading_factor):
        """
        Apply MDSSS mapping to the given binary signal

        Parameters
        ----------
        message : ndarray
            Input bitstream
        spreading_factor : int
            MDSSS Spreading factor 

        Returns
        -------
        output : ndarray
            Output bitstream

        """
        N = spreading_factor // 4
        self._print_verbose(f"TPC N = {N}")
        bitstreams = self._TPC(message, N)
        self._print_verbose(f"TPC Encoding {message.shape} -> {bitstreams.shape}")

        # Call the spreading and multiplexing function
        output = self._spreading_multiplexing(bitstreams, N)

        return output

    def _SHR(self):
        """
        Returns SHR bitstream

        Returns
        -------
        signal : ndarray
            SHR bitstream
        """

        signal = np.concatenate([np.zeros(
            PREAMPLE_LENGTH[self._frequency_band], dtype=np.uint8), SFD], dtype=np.uint8)
        return signal

    def _PHR(self, message_length):
        """
        Returns PHR bitstream

        Parameters
        ----------
        message_length : int
            Length of the PSDU (prior to FEC encoding) in octets


        Returns
        -------
        signal : ndarray
            SHR bitstream
        """

        signal = np.zeros([PHR_LENGTH * 8], dtype=np.uint8)
        # See Figure 114
        signal[0] = 1 if self._SM else 0  # SM
        signal[1:2+1] = to_bin(self._RM, 2)  # Rate Mode
        signal[3:4+1] = 0  # Reserved
        signal[5:15+1] = to_bin(message_length, 11)  # Frame length

        # Calculate HCS

        # 0000000000000111 -> 00010101

        # generator polynomial (x⁸ + x² + x + 1)
        G8 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.uint8)
        x8 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        # Signal to apply HCS to
        M = signal[:16]
        # Multiply M by x⁸
        M2 = np.mod(np.polymul(M, x8).astype(np.uint8), 2)
        # Divide and get the remainder
        R = np.mod(np.polydiv(M2, G8)[1].astype(np.uint8), 2)
        # Pad the remainder with zeros
        HCS = np.concatenate([np.zeros(8 - R.size), R])
        signal[16:] = HCS

        return signal

    def _insert_pilots(self, encoded_PSDU):
        """
        Insert pilots into encoded PSDU

        Parameters
        ----------
        encoded_PSDU : ndarray
            PSDU after encoding (DSSS or MDSSS)
        
        Returns
        -------
        output : ndarray
            PSDU with inserted pilots
        """
        MP = PILOT_SPACING[self._frequency_band]
        pilot_sequence = PILOT_SEQUENCES[self._frequency_band]

        NP = pilot_sequence.size

        L = int(np.ceil(encoded_PSDU.size / MP))
        output = np.zeros(L*NP + encoded_PSDU.size, dtype=encoded_PSDU.dtype)

        # segment length (pilot length + psdu length)
        sl = NP + MP
        for l in range(L):
            output[l*sl:(l+1)*sl] = np.concatenate([pilot_sequence,
                                                    encoded_PSDU[l*MP:(l+1)*MP]])

        return output

    def _FEC(self, data, isPSDU):
        """
        Apply FEC encoding to the data (PHR + PSDU) and appends tail and pad bits

        Parameters
        ----------
        data : ndarray
            Bitstream of the message to encode
        isPSDU : bool
            True if the data is the PSDU, False otherwise

        Returns
        -------
        output : ndarray
            Output bitstream after FEC encoding    
        """
        if isPSDU:
            N_INTRLV = 7 * 18 # See interleaver
            LENGTH = data.size // 8
            NB = np.ceil((8 * LENGTH + 6) / (N_INTRLV / 2))
            ND = NB * (N_INTRLV / 2)
            N_PAD = int(ND - (8 * LENGTH + 6))


            # Add 6 zeros and PAD
            data_pad = np.concatenate([data, np.zeros(6 + N_PAD, dtype=np.uint8)])
        else:
            # Add 6 zeros to the data
            data_pad = np.concatenate(
                [data, np.zeros(6, dtype=np.uint8)], dtype=np.uint8)

        def iter(u, u_new):
            ak0 = u_new ^ ((u >> 4) & 1) ^ (
                (u >> 3) & 1) ^ ((u >> 1) & 1) ^ (u & 1)
            ak1 = u_new ^ ((u >> 5) & 1) ^ (
                (u >> 4) & 1) ^ ((u >> 3) & 1) ^ (u & 1)

            u_next = u >> 1 | (u_new << 5)

            return u_next, ak0, ak1

        ak = []
        u = 0

        for u_new in data_pad:
            u, ak0, ak1 = iter(u, u_new)
            ak.append(ak0)
            ak.append(ak1)

        return np.array(ak)

    def _interleaver(self, data, isPSDU):
        """
        Applies interleaver to the data

        Parameters
        ----------
        data : ndarray
            Input message (bitstream)
        isPSDU : bool
            True if the data is the PSDU, False otherwise

        Returns
        -------
        output : ndarray
            Output bitstream after interleaving
        """
        # Degree
        if isPSDU:
            Lambda = 7
            N_INTRLV = 7 * 18
        else:
            Lambda = 6
            N_INTRLV = 10 * 6

        k = np.arange(data.size)
        i = (N_INTRLV / Lambda * np.mod(N_INTRLV - 1 - k, Lambda) +
             np.floor((N_INTRLV - 1 - k) / Lambda)).astype(int)
        output = np.zeros_like(data)

        output = data[i]

        return output

    # def _message_to_bits(self, byte_message):
    #     """
    #     Check message and convert to bitstream

    #     Parameters
    #     ----------
    #     byte_message : bytearray, ndarray
    #         Message to convert (bytearray or unsigned int array)

    #     Returns
    #     -------
    #     bitstream : ndarray
    #         Output signal (ndarray of 0 or 1)
    #     """
    #     if isinstance(byte_message, np.ndarray):
    #         if np.issubdtype(byte_message, np.integer):
    #             if byte_message.min() < 0 or byte_message.max() > 255:
    #                 raise ValueError(
    #                     "Invalid byte_message range. It should be between 0 and 255")
    #         else:
    #             raise TypeError(
    #                 "byte_message dtype is invalid. It should be integer")
    #     elif not (isinstance(byte_message, bytearray) or isinstance(byte_message, bytes)):
    #         raise TypeError(
    #             "Invalid byte_messsage type. It should be bytearray or a ndarray of integers")
    #     else:
    #         byte_message = np.array(list(byte_message), dtype=np.uint8)

    #     bitstream = np.unpackbits(byte_message, bitorder='big')

    #     return bitstream

    def _o_qpsk_modulator(self, cPPDU, samples_per_symbol):
        """
        Apply O-QPSK modulation to the cPPDU

        Parameters
        ----------
        cPPDU : ndarray
            Input bitstream
        samples_per_symbol : int
            Number of samples per symbol

        Returns
        -------
        y : ndarray
            Output signal
        f : float
            Output signal sampling frequency
        """
        # Chip duration
        bpskMod = BPSK()
        

        chip_rate = DATA_RATE[self._SM][self._frequency_band][self._RM] # bit / s
        Tc = 1 / chip_rate

        self._print_verbose(f"Chip rate : {chip_rate/1e3:.2f} kchip/s")

        # BPSK, complex mapped, spread signal
        self._s = np.zeros(cPPDU.size * samples_per_symbol // 2, dtype=complex)
        self._s[::samples_per_symbol] = bpskMod.convert(cPPDU[::2]) + 1j * bpskMod.convert(cPPDU[1::2])
        self._s_t = np.arange(self._s.size) * Tc
        
        if self._frequency_band in [Frequency_band.Band_915MHz, Frequency_band.Band_2450MHz]:
            p = half_sine
        else:           
            p = raised_cosine

        # Pulse signal
        self._pulse_t = np.linspace(-2*Tc, 2*Tc, samples_per_symbol, endpoint = False)
        self._pulse = p(self._pulse_t, Tc)
        
        # Convolution of the spread s signal and pulse signal
        output = np.convolve(self._s, self._pulse)

        f = chip_rate * samples_per_symbol

        return output, f

    def bitsToIQ(self, bits):
        """
        Encode a binary message (list of bits) with MR-O-QPSK modulator

        Parameters
        ----------
        bits : ndarray or list
            List of bits to send (PSDU)

        Returns
        -------
        signal : ndarray
            output bitstream
        f : float
            signal frequency        
        """
        bits = check_binary_array(bits)

        # Reference : Figure 144
        # SHR
        shr = self._SHR()
        self._shr_bde = self._BDE(shr, 0)
        self._cSHR = self._DSSS(
            self._shr_bde, spreading=SHR_SPREADING[self._frequency_band])
        # PHR
        phr = self._PHR(bits.size // 8)
        self._phr_encoded = self._FEC(phr, isPSDU=False)
        self._phr_interleaved = self._interleaver(
            self._phr_encoded, isPSDU=False)
        self._phr_bde = self._BDE(
            self._phr_interleaved, Eprevious=self._shr_bde[-1])
        self._cPHR = self._DSSS(
            self._phr_bde, PHR_SPREADING[self._frequency_band])
        # PSDU
        # FEC + interleaving always active with DSSS, only active with Spreading
        # mode is 0 or 1 with MDSSS
        if 0 <= self._RM <= 1 or self._SM == SpreadingMode.DSSS.value: 
            self._PSDU_encoded = self._FEC(bits, isPSDU=True)
            self._PSDU_interleaved = self._interleaver(self._PSDU_encoded, isPSDU=True)
            self._PSDU_temp = self._PSDU_interleaved
        else:
            self._PSDU_temp = bits
        

        if self._SM == SpreadingMode.DSSS.value:
            if PSDU_BDE_DSSS[self._frequency_band][self._RM]:
                # BDE enabled
                self._PSDU_BDE = self._BDE(self._PSDU_temp, Eprevious=0)
            else:
                # BDE disabled
                self._PSDU_BDE = self._PSDU_temp
            
            if PSDU_SPREADING_DSSS[self._frequency_band][self._RM] != False:
                # DSSS Enabled
                PSDU_encoded_spread = self._DSSS(
                    self._PSDU_BDE, spreading=PSDU_SPREADING_DSSS[self._frequency_band][self._RM])
            else:
                PSDU_encoded_spread = self._PSDU_BDE
        else:
            PSDU_encoded_spread = self._MDSSS(self._PSDU_temp, spreading_factor=PSDU_SPREADING_FACTOR_MDSSS[self._frequency_band][self._RM])

        self._cPSDU = self._insert_pilots(PSDU_encoded_spread)
        # Concatenate SHR + PHR + PSDU
        self._cPPDU = np.concatenate([
            self._cSHR,
            self._cPHR,
            self._cPSDU
        ])
        # Apply O-QPSK modulation
        signal, f = self._o_qpsk_modulator(self._cPPDU, self._samples_per_symbol)
        #signal = 0
        #f = 0
 
        return signal, f

    def bytesToIQ(self, bytes):
        """
        Encode a byte-array message (list of bytes) with MR-O-QPSK modulator

        Parameters
        ----------
        bytes : ndarray or list or bytes
            List of bytes to send (PSDU)
        
        Returns
        -------
        signal : ndarray
            output bitstream
        f : float
            signal frequency

        """
        # Convert to bits
        message_bin = to_binary_array(bytes)
        #message_bin = self._message_to_bits(message)
        return self.bitsToIQ(message_bin)        

    def _print_verbose(self, message: str):
        """
        Prints additionnal information if the verbose flag is True
        """
        if(self._verbose):
            print(message)