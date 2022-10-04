"""
Functions to implement modulations used for OFDM

Modulation    M
BPSK          2
QPSK          4
16-QAM        16
"""
import numpy as np


class Modulator():
    def __init__(self, MSB_first=True, axis=0):
        """
        Modulator class

        Parameters
        ----------
        MSB_first : bool
            Describes if MSB is first in the array (default True). Ignored for BPSK modulation
        axis : int
            Specifies in which axis the conversion must be done (for 2D arrays)
        """
        # Check types and values
        if not isinstance(MSB_first, bool):
            raise ValueError('MSB_first must be of type bool')
        if not isinstance(axis, int):
            raise ValueError('axis must be of type int')
        elif axis < 0:
            raise ValueError('axis cannot be negative')

        self.MSB_first = MSB_first
        self.axis = axis
        self.name = ""

    def convert(self, signal):
        """
        Applies modulation to the signal

        Parameters
        ----------
        signal : ndarray
            Input signal (bits)

        Returns
        -------
        output : ndarray
            Modulated output signal values
        """
        raise NotImplementedError("modulator class cannot be used directly")

    def bits_per_symbol(self):
        raise NotImplementedError("modulator class cannot be used directly")


class BPSK(Modulator):
    """
    Maps >0 values to 1 and everything else to -1
    """

    def convert(self, signal):
        HIGH = 1 + 0j
        if isinstance(signal, int) or isinstance(signal, float):
            # Single value
            output = HIGH if signal > 0 else -HIGH
        elif isinstance(signal, np.ndarray):
            # Array
            output = np.ones_like(signal) * (-HIGH)
            output[signal > 0] = HIGH
        elif isinstance(signal, list):
            output = [HIGH if a > 0 else -HIGH for a in signal]
        else:
            raise ValueError("Unsupported type")
        return output

    def bits_per_symbol(self):
        return 1


class QPSK(Modulator):
    """
    Separates the signal by group of 2 bits then Maps the signal on a square
        00 -> -1-1j
        01 -> -1+1j
        10 -> +1-1j
        11 -> +1+1j
    """

    def convert(self, signal):
        HIGH = 1
        if isinstance(signal, np.ndarray):
            BPSK_mod = BPSK(MSB_first=self.MSB_first, axis=self.axis)

            if self.MSB_first:
                first, second = 1, 1j
            else:
                first, second = 1j, 1

            if signal.ndim == 1:
                # One-dimensional array
                output = (BPSK_mod.convert(signal[::2]) * first +
                            BPSK_mod.convert(signal[1::2]) * second) * HIGH
            elif signal.ndim == 2:
                # 2D array (mapping along the specified axis)
                if self.axis == 0:
                    assert np.mod(
                        signal.shape[0], 2) == 0, "Invalid number of elements"
                    output = (BPSK_mod.convert(signal[::2, :]) * first +
                                BPSK_mod.convert(signal[1::2, :]) * second) * HIGH
                elif self.axis == 1:
                    assert np.mod(
                        signal.shape[1], 2) == 0, "Invalid number of elements"
                    output = (BPSK_mod.convert(signal[:, ::2]) * first +
                                BPSK_mod.convert(signal[:, 1::2]) * second) * HIGH
                else:
                    raise ValueError("Invalid axis")
            else:
                raise ValueError(f"Invalid signal shape ({signal.shape})")
            return output
        else:
            raise ValueError("Unsupported type")

    def bits_per_symbol(self):
        return 2


class QAM16(Modulator):
    """
    Maps the values on a 4x4 QAM matrix
    (multiple standards exists for the symbols order)
        -3j    -j    +j    +3j
        1000  1001  1011  1010 +3
        1100  1101  1111  1110 +1
        0100  0101  0111  0110 -1
        0000  0001  0011  0010 -3

    """

    def convert(self, signal):
        # Above/Below x axis is given by MSB
        # Vertically "close" to center is given by the next bit
        # Right/Left of Y axis is given by the next bit
        # Horizontally "close" to center is given by LSB
        axis = self.axis

        if(isinstance(signal, np.ndarray)):
            # We expect the signal to be a numpy array
            if signal.ndim == 1:
                # Make it a column vector
                signal = signal.reshape(-1, 1)
                axis = 0  # force the axis
            elif signal.ndim == 2:
                if axis == 1:
                    # Swap the signal
                    signal = signal.T
                elif axis != 0:
                    raise ValueError(f"Invalid axis ({axis})")
            else:
                raise ValueError(f"Invalid signal shape {signal.shape}")

            BPSK_mod = BPSK(axis=0)

            # Imaginary part
            imag = BPSK_mod.convert(signal[(None if self.MSB_first else 3)::4, :]) * 2 * 1j * (BPSK_mod.convert(
                signal[(1 if self.MSB_first else 2)::4, :]) / 2 + 1)
            # Real part
            real = BPSK_mod.convert(signal[(2 if self.MSB_first else 1)::4, :]) * 2 * (BPSK_mod.convert(
                signal[(3 if self.MSB_first else None)::4, :]) /2 + 1)

            output = real + imag

            return output

        else:
            raise ValueError("Unsupported type")

    def bits_per_symbol(self):
        return 4


def get_modulator_dict():
    """
    Returns the list of possible modulators
    """
    return {'BPSK': BPSK, 'QPSK': QPSK, 'QAM16': QAM16}


def get_modulator(mod: str, MSB_first=True, axis=0) -> Modulator:
    """
    Provides the requested modulator as subclass of modulator()

    Parameters
    ----------
    mod : str
        The requested modulator as a string

    get_modulator_list() gives the list of acceptable values
    """
    if not mod in get_modulator_dict().keys():
        raise ValueError("Invalid modulator")
    else:
        mod_instance = get_modulator_dict()[mod](MSB_first, axis)
        mod_instance.name = mod
        return mod_instance
