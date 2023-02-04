"""
Main OFDM modulator class

Must be instanciated as such :
    modulator = ofdm_modulator(settings)

Output can be requested by calling the appropriate function, i.e.
    I, Q = modulator.getIQ()
"""

from ..tools.modulations import get_modulator
from ..tools.pn9 import Pn9
import math
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq


class Ofdm_modulator():
    def __init__(self, N_FFT, modulation, modulation_factor, CP,
                 padding_left, padding_right, frequency_spreading=1, pilots_indices=None, pilots_values=None, 
                 initial_pilot_set=0, initial_pn9_seed=0x1FF, verbose=False):
        """
        Returns an OFDM modulator with the desired settings

        Parameters
        ----------
        N_FFT : int
            Number of FFT channels used (default 32)
        modulation : {'BPSK', 'QPSK', 'QAM16'}
            Type of modulation used, by default BPSK
        modulation_factor : float
            Factor to multiply the modulation with
        CP : float
            Cyclic prefix. This fraction ot the end of the symbol in the time domain is repeated at the beginning
        padding_left : int
            Empty FFT channels on the left (negative frequencies)
        padding_right : int
            Empty FFT channels on the right (positive frequencies)
        frequency_spreading : int
            1 : no spreading
            2 : 2x spreading (half the data rate)
            4 : 4x spreading (1/4 the data rate)
            See 18.2.3.6 Frequency spreading
        pilots_indices : numpy array
            Indices of pilots in the OFDM symbol.
            If 1D : pilots positions are constant throughout the symbols.
            if 2D :each column represents a new symbol. Looping will occur when all the column have been used
            None by default (no pilots).
            For a 16 FFT :
            0 is the center frequency (DC)
            -8 is the lowest index
            +7 is the highest index
        pilots_values : numpy array or str
            Pilots values.
            If 1D array     : sequence of pilots (restarts when the end is reached). A single value in the array is valid
            If "pn9"        : use a pseudo-random (PN9) sequence to generate pilots values
            None by default (no pilots)
        initial_pilot_set : int
            If multiple pilot sets are specified, this value is the starting one (defaults to 0)
        initial_pn9_seed : int
            Initial value for the pn9 sequence (if used)
        verbose : bool
            if True, prints informations throughout the process (False by default)
        """
        # Check types and values
        if not isinstance(N_FFT, int):
            raise ValueError("N_FFT must be int type")
        else:
            if not math.log(N_FFT, 2).is_integer():
                raise ValueError("N_FFT must be a power of 2")
        if not isinstance(modulation, str):
            raise ValueError("modulation must be str")
            # No need to check for valid string, the modulation module will do it
        if not isinstance(padding_left, int):
            raise ValueError("padding_left must be int")
        if not isinstance(padding_right, int):
            raise ValueError("padding_right must be int")
        if pilots_indices is None and pilots_values is not None:
            raise ValueError("Pilots must have corresponding indices")
        elif pilots_indices is not None and pilots_values is None:
            raise ValueError("Pilots indices must have pilots values")
        if not isinstance(frequency_spreading, int):
            raise ValueError("frequency_spreading must be int")
        elif not frequency_spreading in [1, 2, 4]:
            raise ValueError(
                f"Invalid frequency_spreading value {frequency_spreading}, must be [1, 2, 4]")
        if not isinstance(initial_pilot_set, int):
            raise ValueError("Invalid initial_pilot_set type")
        if not isinstance(initial_pn9_seed, int):
            raise ValueError("Invalid initial_pn9_seed type")

        # Manage pilots
        if pilots_values is not None:
            # Check for types
            if isinstance(pilots_indices, list):
                pilots_indices = np.asarray(pilots_indices)
            elif not isinstance(pilots_indices, np.ndarray):
                raise ValueError("pilots_indices must be a numpy array or a list")
            if not np.all(np.diff(pilots_indices, axis=0) >= 0):
                raise ValueError("Pilots indices must be ordered")
            if isinstance(pilots_values, list):
                pilots_values = np.asarray(pilots_values)
            if not (isinstance(pilots_values, np.ndarray) or isinstance(pilots_values, str)):
                raise ValueError(
                    "Invalid pilots values type (must be numpy array or str)")
            if not (1 <= pilots_indices.ndim <= 2):
                raise ValueError(
                    f"Invalid pilots indices shape ({pilots_indices.shape})")

            self._N_pilots = pilots_indices.shape[0]

            if isinstance(pilots_values, str) and pilots_values == "pn9":
                # Use PN9 sequence
                self._use_pn9_sequence = True
            else:
                self._use_pn9_sequence = False
                # Check if an array is given for pilots values and it matches the size of pilots_indices
                if pilots_values.ndim > 0 and pilots_values.shape[0] != pilots_indices.shape[0]:
                    raise ValueError(
                        f"pilots_indices and pilots_values do not match ({pilots_values.shape[0]} != {pilots_indices.shape[0]})")
            # Store the values
            self._pilots_values = pilots_values
            # Reshape to 2D (so that the rows always represent the pilots positions)
            if pilots_indices.ndim == 1:
                self._pilots_indices = pilots_indices.reshape(-1, 1)
            else:
                self._pilots_indices = pilots_indices
        else:
            self._N_pilots = 0

        self._pseudo_random_sequence = Pn9(initial_pn9_seed)

        # Save the values in the class
        self._N_FFT = N_FFT
        self._modulator = get_modulator(modulation)
        self._verbose = verbose
        self._padding_left = padding_left
        self._padding_right = padding_right
        self._modulation_factor = modulation_factor
        self._frequency_spreading = frequency_spreading

        self._CP = CP

        self._N_bpsc = self._modulator.bits_per_symbol()

        self._DC_TONE = 1
        # Length is FFT minus all the non-data channels (padding + pilots)
        self._message_split_length = int((
            self._N_FFT - self._N_pilots - self._padding_left - self._padding_right - self._DC_TONE) * self._N_bpsc / self._frequency_spreading)

        # Used only with variable pilots indices (multiple columns)
        self._pilots_column_index = initial_pilot_set

    def _split_message(self, message, pad=False):
        """
        Splits a message into multiple parts (each with a size suitable for the IFFT)

        Parameters
        ----------
        message : 1D numpy.array
            The message to modulate. Size must be a multiple of (N_FFT - number of pilots)
        pad : bool
            Pad the message with 0s to reach the desired length. False by default
        """
        # We assume the message is 1D and with the correct number of elements
        #
        # The split order is :
        #
        # 0  n+1 . .
        # 1  n+2 . .
        # .   .  . .
        # .   .  . .
        # n  2n  . .
        #
        # Each column represents an OFDM symbol
        # Each element (row by row) is a IFFT channel (minus the pilots)
        self._print_verbose("Splitting message...")
        if pad:
            # Add 0s to reach the right size
            old_message_shape = message.shape
            missing_zeros = int(np.ceil(
                message.size / self._message_split_length) * self._message_split_length - message.size)
            message = np.concatenate([message, np.zeros(missing_zeros)])
            self._print_verbose(
                f"    Padding message with {missing_zeros} zeros ({old_message_shape} -> {message.shape})")

        message_split = message.reshape(
            self._message_split_length, -1, order='F')
        self._print_verbose(
            f"    Splitting message from length {message.size} to {message_split.shape} ({message_split.shape[0]} channels before mapping and {message_split.shape[1]} OFDM symbols)")
        return message_split

    def _constellation_map(self, message):
        """
        Maps the message with the specified constellation

        Parameters
        ----------
        message : numpy array
            Values to map along rows. Each column is a symbol 

        Returns
        -------
            mapped_message : numpy array
                Converted message
        """
        self._print_verbose(
            f"Constellation mapping using {self._modulator.name}...")
        self._print_verbose(
            f"    Modulation factor is {self._modulation_factor}...")
        mapped_message = self._modulator.convert(
            message) * self._modulation_factor
        self._print_verbose(
            f"    New message size {message.shape} -> {mapped_message.shape} ({mapped_message.shape[0]} + pilots as IFFT channels and {mapped_message.shape[1]} OFDM symbols)")
        return mapped_message

    def _frequency_spread(self, message):
        """
        Applies frequency spreading to the message (repetition)
        1 : No repetition
        2 : Data is placed in the positive values of the FFT and copied on the lower (negative) values
        4 : Data is placed in the lower positive values of the FFT and copied on the 4 others segments
        See 18.2.3.6 Frequency spreading
        """

        if self._frequency_spreading == 1:
            self._print_verbose("Frequency spreading of 1 (no modification)")
            # No spreading to do
            return message
        elif self._frequency_spreading == 2:
            self._print_verbose("Frequency spreading of 2 (2x repetition)")
            # d(k-Nd/2-1) = d(k) * e^(j*2*pi*(2*k-1)/4)

            # k is the index (positive and starts at 1)
            # k :
            #  0 0 0 ..
            #  1 1 1 ..
            #  2 2 2 ..
            #  . . .
            k = (
                np.arange(message.shape[0])+1).repeat(message.shape[1]).reshape(*message.shape)
            # phase is the phase matrix (for each data value)
            phase = np.exp(1j*2*np.pi*(2*k-1)/4)
            # lower portion of the FFT, message is the higher portion
            lower = message * phase

            spread_message = np.block([[lower], [message]])

            return spread_message
        elif self._frequency_spreading == 4:
            self._print_verbose("Frequency spreading of 4 (4x repetition)")
            # Same principle as 2x but a bit more complicated
            # message is located here :
            # ----------------Dmmmmmmmm--------
            # Setting k matrix (just like before)
            k = (
                np.arange(message.shape[0])+1).repeat(message.shape[1]).reshape(*message.shape)
            # ----------------D--------xxxxxxxx
            phase = np.exp(1j*2*np.pi*(k-1)/4)
            positive_high = message * phase
            # xxxxxxxx--------D----------------
            phase = np.exp(1j*2*np.pi*(2*k - 1)/4)
            negative_low = message * phase
            # --------xxxxxxxxD----------------
            phase = np.exp(1j*2*np.pi*(3*k - 1)/4)
            negative_high = message * phase

            spread_message = np.block(
                [[negative_low], [negative_high], [message], [positive_high]])
            return spread_message
        else:
            raise ValueError("Invali frequency spreading factor")

    def _add_pilots_and_padding(self, message):
        """
        Adds pilots and paddings to the message (to reach IFFT size)

        Parameters
        ----------
        message : numpy array (rows are IFFT channels and columns are symbols)

        Returns
        -------
        ifft_channels : numpy array (same format as message)        
        """
        message_str = ['-']*self._N_FFT
        self._print_verbose(
            "Adding OFDM " + ("pilots, " if self._N_pilots else '') + "DC Tone and padding...")
        self._print_verbose("  Message without pilots :")
        self._print_verbose(f"    {''.join(message_str)} ({message.shape[0]}x)")

        # Create the new signal matrix (empty for now)
        ifft_channels = np.zeros(
            [self._N_FFT, message.shape[1]], dtype=complex)
        available_channels_global = list(
            range(-self._N_FFT//2, self._N_FFT//2))
        # Adding padding
        self._print_verbose(
            f"  Adding padding ({self._padding_left}, {self._padding_right})")
        if self._padding_left > 0:
            message_str[0:self._padding_left] = ['0'] * self._padding_left
            available_channels_global = available_channels_global[self._padding_left:]
        if self._padding_right > 0:
            message_str[-self._padding_right:] = ['0'] * self._padding_right
            available_channels_global = available_channels_global[:-self._padding_right]
        # No need to set the ifft_channels to zero for padding since there are already 0
        self._print_verbose(f"    {''.join(message_str)} ({message.shape[0]}x)")

        # Adding DC tone
        self._print_verbose("  Adding DC Tone")
        message_str[len(message_str)//2] = 'D'
        self._print_verbose(f"    {''.join(message_str)} ({ifft_channels.shape[0]}x)")
        available_channels_global.remove(0)  # Remove value=0
        # Again, no need to set the value to 0

        # Create a BPSK modulator for PN9 sequence (if used)
        bpsk_modulator = get_modulator('BPSK')

        self._pilots_values_index = 0

        # Adding pilots
        if self._N_pilots > 0:
            # Iterate over the symbols
            for c in range(message.shape[1]):
                message_str_i = message_str.copy()
                symbol = message[:, c]
                available_channels = available_channels_global.copy()  # For this symbol

                pilot_set = self._pilots_indices[:, self._pilots_column_index]
                self._print_verbose(
                    f"    Adding pilots at {pilot_set} (set {self._pilots_column_index})")

                # use the pilots_column_index to select which column to set (stored in the class)
                for row_index in self._pilots_indices[:, self._pilots_column_index] + self._N_FFT//2:
                    if self._use_pn9_sequence:
                        # PN9 sequence
                        p = bpsk_modulator.convert(
                            np.array([self._pseudo_random_sequence.next()]))[0]
                    else:
                        # Array
                        p = self._pilots_values[self._pilots_values_index]
                        self._pilots_values_index += 1
                        if self._pilots_values_index >= self._pilots_values.size:
                            self._pilots_values_index = 0
                    # Insert pilot at row (row is 0..N_FFT-1)
                    ifft_channels[row_index, c] = p
                    available_channels.remove(row_index - self._N_FFT//2)

                    # Visual stuff :
                    message_str_i[row_index] = 'P'

                self._pilots_column_index += 1
                if self._pilots_column_index >= self._pilots_indices.shape[1]:
                    self._pilots_column_index = 0

                    self._print_verbose(
                        f"    {''.join(message_str_i)} ({ifft_channels.shape[0]}x) (pilot set/index : {self._pilots_column_index})")

                # Insert data where there's room
                ifft_channels[[a + self._N_FFT //
                               2 for a in available_channels], c] = message[:, c]
        else:
            ifft_channels[[a + self._N_FFT //2 for a in available_channels_global],:] = message


        return ifft_channels

    def _ifft(self, channels):
        """
        Applies iFFT to the message (channels). The iFFT is applied on the rows (each column is a separate symbol)

        Parameters
        ----------
        channels : numpy array
            The input data of the iFFT

        Returns
        -------
        signal : numpy array
            time domain signal
        """
        self._print_verbose(f"Applying iFFT to the signal...")
        # ifftshift is very important since the spectrum was created "how it looks" but the ifft does 0-> Fs/2 -> -Fs/2 -> 0-dF
        self._subcarriers = channels.squeeze()
        signal = ifft(ifftshift(channels, axes=0), axis=0, norm='ortho')

        return signal

    def _cyclic_prefix(self, signal):
        """
        Adds cyclic prefix to the corresponding signal (along the rows). The fraction of cyclic prefix is given by IG

        Parameters
        ----------
        signal : numpy array
            the signal

        Returns
        -------
        cyclic_signal : numpy array
            the signal with cyclic prefix
        """
        if(self._CP > 0):
            self._print_verbose("Adding cyclic prefix...")
            prefix_length = int(signal.shape[0]*self._CP)
            prefix = signal[-prefix_length:, :]
            cyclic_signal = np.concatenate([prefix, signal])
            self._print_verbose(
                f"    Signal {signal.shape} -> {cyclic_signal.shape}")
            return cyclic_signal
        else:
            return signal

    def bitsToIQ(self, bits, pad=False):
        """
        Applies OFDM modulation to the provided message

        Parameters
        ----------
        bits : numpy array
            Array containing the bits to transmit
        pad : bool
            Pad the message with 0s to reach the desired length. False by default

        Returns
        -------
        I : ndarray
            Real part of the signal
        Q : ndarray
            Imaginary part of the signal
        """
        # Convert list to numpy array if necessary
        if isinstance(bits, list):
            bits = np.asarray(bits)
        # Check types and values
        if not isinstance(bits, np.ndarray):
            raise ValueError("Message must be a numpy array")
        bits = np.squeeze(bits)
        if bits.ndim != 1:
            raise ValueError("Message must be one-dimensional")
        elif np.mod(bits.size, self._message_split_length) != 0 and pad == False:
            raise ValueError(
                f"Message size must be a multiple of {self._message_split_length} (it currently is {bits.size})")

        ### Splitting (separating message into OFDM symbols before IFFT) ###
        message_split = self._split_message(bits, pad)

        ### Constellation mapping ###
        message_split_mapped = self._constellation_map(message_split)

        ### Frequency spreading ###
        message_split_mapped_spread = self._frequency_spread(
            message_split_mapped)

        ### Adding pilots and padding###
        message_split_mapped_pilots = self._add_pilots_and_padding(
            message_split_mapped_spread)

        ### IFFT ###
        signal = self._ifft(message_split_mapped_pilots)

        ### Cyclic prefix ###
        signal_cyclic = self._cyclic_prefix(signal)

        ### Reshape output to a single vector ###
        return signal_cyclic.reshape(-1, order='F')

    def subcarriersToIQ(self, subcarriers):
        """
        Converts a list of subcarriers to I and Q signals
        Essentially this function applies IFFT and cyclic prefix

        Parameters
        ----------
        subcarriers : ndarray
            List of subcarriers. 1D or 2D with each column corresponding to a symbol.
            Must have the same number of samples as N_FFT

        Returns
        -------
            output : ndarray
                Output complex signal
        """
        if subcarriers.ndim == 1:
            subcarriers = subcarriers.reshape(-1, 1)
        elif subcarriers.ndim != 2:
            raise ValueError(
                f"Invalid number of dimensions ({subcarriers.ndim})")

        if subcarriers.shape[0] != self._N_FFT:
            raise ValueError(
                f"Invalid number of subcarriers ({subcarriers.shape[0]} / {self._N_FFT})")

        ### IFFT ###
        signal = self._ifft(subcarriers)

        ### Cyclic prefix ###
        signal_cyclic = self._cyclic_prefix(signal)

        return signal_cyclic

    def get_pilot_set_index(self):
        """
        Returns the current pilot set index (to continue the message)
        The index starts at 0

        Returns
        -------
        pilot_set_index : int
            The index of the NEXT pilot set to use
        pn_seed : int
            Seed for the pseudo-random sequence
        """
        return self._pilots_column_index

    def get_pn9_value(self):
        """
        Returns the current value of the pn9 generator

        Returns
        -------
        value : int
            Value of the pseudo-random generator for pilots generation
        """
        return self._pseudo_random_sequence.get_current_value()

    def get_number_of_bits_per_symbol(self):
        """
        Return the number of bits used for each symbol (before spreading and modulation)

        Returns
        -------
        output : int
            Number of bits per symbol
        """
        return self._message_split_length
        
    def _print_verbose(self, message: str):
        """
        Prints additionnal information if the verbose flag is True
        """
        if(self._verbose):
            print(message)
