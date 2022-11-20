"""
MR-FSK Modulator
"""

import numpy as np
from enum import Enum
from ..tools.bits import from_bitstring, to_binary_array
from ..tools import operations

from colorama import Fore


class Modulation(Enum):
    FSK2 = 1
    FSK4 = 2


# SFD Values as a function of modulation, phyMRFSKSFD and coded/uncoded
# Key is (modulation, phyMRFSKSFD, is_coded)
# See tables 131 and 132 of 802.15.4g-2012
SFD = {
    # Table 131
    (Modulation.FSK2, 0, True): from_bitstring('0110 1111 0100 1110'),
    (Modulation.FSK2, 0, False): from_bitstring('1001 0000 0100 1110'),
    (Modulation.FSK2, 1, True): from_bitstring('0110 0011 0010 1101'),
    (Modulation.FSK2, 1, False): from_bitstring('0111 1010 0000 1110'),
    # Table 132
    (Modulation.FSK4, 0, True):  from_bitstring('0111 1101 1111 1111 0111 0101 1111 1101'),
    (Modulation.FSK4, 0, False): from_bitstring('1101 0111 0101 0101 0111 0101 1111 1101'),
    (Modulation.FSK4, 1, True):  from_bitstring('0111 1101 0101 1111 0101 1101 1111 0111'),
    (Modulation.FSK4, 1, False): from_bitstring('0111 1111 1101 1101 0101 0101 1111 1101'),
}

# preamble field as a function of modulation
PREAMBLE_SEQUENCE = {
    Modulation.FSK2: np.tile(np.array([0, 1]), 4),
    Modulation.FSK4: np.tile(np.array([0, 1, 1, 1]), 4)

}

PHR_LENGTH = 2

# Tail bits based on the memory state of the RSC encoder
RSC_TAIL_BITS = {
    0b000: np.array([0, 0, 0]),
    0b001: np.array([1, 0, 0]),
    0b010: np.array([1, 1, 0]),
    0b011: np.array([0, 1, 0]),
    0b100: np.array([1, 1, 1]),
    0b101: np.array([0, 1, 1]),
    0b110: np.array([0, 0, 1]),
    0b111: np.array([1, 0, 1]),
}
# Tail bits for the NRNSC encoder
NRNSC_TAIL_BITS = np.array([0, 0, 0])


class Mr_fsk_modulator:
    def __init__(self, phyMRFSKSFD, phyFSKPreambleLength, modulation, phyFSKFECEnabled, phyFSKFECScheme, macFCSType, phyFSKScramblePSDU, phyFSKFECInterleavingRSC):
        """
        Creates an instance of a MR-FSK modulator

        Parameters
        ----------
        phyMRFSKSFD : int
            Selection of the SFD group (See table 131 of 802.15.4g)
        phyFSKPreambleLength : int
            Length of the preamble
        modulation : str
            Modulation type : "2FSK" or "4FSK"
        phyFSKFECEnabled : bool
            Enable FEC encoding (True) or not (False)
        phyFSKFECScheme : int
            Configures the FEC mode. 0 for NRNSC and 1 for RSC
        macFCSType : int
            Lengths of the FCS 0 -> 4, 1 -> 2
            FCS Type describing the length of transmitted FCS.
        phyFSKScramblePSDU : bool
            Enable (True) or disable (False) the whitening of the PSDU
        phyFSKFECInterleavingRSC : bool
            Enable (True) interleaving for RSC or disable (False)

        """
        # Checks
        if isinstance(phyMRFSKSFD, int):
            if phyMRFSKSFD not in [0, 1]:
                raise ValueError("phyMRFSKSFD should be 0 or 1")
        else:
            raise TypeError("Invalid phyMRFSKSFD type. It should be int")

        if isinstance(phyFSKPreambleLength, int):
            if not (4 <= phyFSKPreambleLength <= 1000):
                raise ValueError(
                    "phyFSKPreambleLength value is invalid. The range is 4-1000 (See Table 71)")
        else:
            raise TypeError("phyFSKPreambleLength must be an integer")

        if isinstance(modulation, str):
            if modulation not in ["2FSK", "4FSK"]:
                raise ValueError(
                    "Invalid modulation type. It should be \"2FSK\" or \"4FSK\"")
        else:
            raise TypeError("Invalid modulation type. It should be str")

        if not isinstance(phyFSKFECEnabled, bool):
            raise TypeError("phyFSKFECEnabled should be of type bool")

        if not isinstance(phyFSKFECScheme, int):
            raise TypeError("phyFSKFECScheme should be of type int")
        elif phyFSKFECScheme not in [0, 1]:
            raise ValueError("phyFSKFECScheme should be 0 or 1")

        if not isinstance(macFCSType, int):
            raise TypeError("FCS_length should be of type int")
        elif macFCSType not in [0, 1]:
            raise ValueError("FCS_length should be 0 or 1")

        if not isinstance(phyFSKScramblePSDU, bool):
            raise TypeError("phyFSKScramblePSDU should be of type bool")

        if not isinstance(phyFSKFECInterleavingRSC, bool):
            raise TypeError("phyFSKFECInterleavingRSC should be of type bool")

        self._macFCSType = macFCSType
        self._phyFSKFECEnabled = phyFSKFECEnabled
        self._phyMRFSKSFD = phyMRFSKSFD
        self._phyFSKPreambleLength = phyFSKPreambleLength
        self._modulation = Modulation.FSK2 if modulation == "2FSK" else Modulation.FSK4
        self._phyFSKScramblePSDU = phyFSKScramblePSDU
        self._phyFSKFECInterleavingRSC = phyFSKFECInterleavingRSC
        self._phyFSKFECScheme = phyFSKFECScheme

    def _bin(self, number, width=8, MSB_first=True):
        """
        Converts a number to binary representation with LSB first

        Parameters
        ----------
        number : int

        Returns
        -------
        output : ndarray
            Array of bits
        """

        return np.array([int(x) for x in np.binary_repr(number, width)[::(1 if MSB_first else -1)]])

    def _SHR(self):
        """
        Returns SHR bitstream

        Returns
        -------
        signal : ndarray
            SHR bitstream
        """

        signal = np.concatenate([
            np.tile(PREAMBLE_SEQUENCE[self._modulation],
                    self._phyFSKPreambleLength),
            SFD[(self._modulation, self._phyMRFSKSFD, self._phyFSKFECEnabled)]
        ]).astype(int)
        return signal

    def _PHR_mode_switch(self, modeSwitchParameterEntry, new_mode_fec, PAGE, MOD, MD):
        """
        Returns a PHR bitstream for mode_switch

        modeSwitchParameterEntry : int
            Mode switch operation (0-3)
        new_mode_fec : bool
            Signal that the packet following is encoded using FEC

        """

        signal = np.zeros([PHR_LENGTH], dtype=int)
        # See figure 115
        signal[0] = 1  # MS
        signal[1:2+1] = self._bin(modeSwitchParameterEntry, 2)
        signal[3] = new_mode_fec  # FEC
        signal[4] = PAGE  # New Mode
        signal[5] = self._bin(MOD, 2)  # New Mode
        signal[7:10+1] = self._bin(MD, 4)  # New Mode
        PC = np.logical_xor.reduce(signal[:10+1])
        # BCH(15,11) code
        g = np.poly1d([1, 0, 0, 1, 1])

        print(Fore.RED + f"Warning : BCH checksum isn't implemented (replaced with zeros)" + Fore.RESET)
        B = np.array([0, 0, 0, 0])
        signal[11:14+1] = self._bin(B, 4)
        signal[15] = PC

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

        signal = np.zeros([PHR_LENGTH * 8], dtype=int)
        # See Figure 114
        signal[0] = 0  # MS
        signal[1:2+1] = 0  # Reserved
        signal[3] = self._macFCSType  # FCS type
        signal[4] = 1 if self._phyFSKScramblePSDU else 0  # DW
        signal[5:] = self._bin(message_length, 11)  # L

        return signal

    def _FEC(self, data):
        """
        Apply FEC encoding to the data (PHR + PSDU) and appends tail and pad bits

        Parameters
        ----------
        data : ndarray
            Bitstream of the message to encode
        """

        def M_iter_RSC(M, bi):
            # Extract bits
            M0, M1, M2 = (M >> 2) & 0b001, (M >> 1) & 0b001, M & 0b001
            # output values
            bi = int(bi)
            ui0 = bi
            ui1 = (bi ^ M0 ^ M1 ^ M2) ^ M1 ^ M2
            # Update M
            M0, M1, M2 = bi ^ M0 ^ M1 ^ M2, M0, M1

            return (M0 << 2) | (M1 << 1) | M2, ui0, ui1

        def M_iter_NRNSC(M, bi):
            # Extract bits
            M0, M1, M2 = (M >> 2) & 0b001, (M >> 1) & 0b001, M & 0b001
            bi = int(bi)
            ui0 = not (bi ^ M0 ^ M1 ^ M2)
            ui1 = not (bi ^ M1 ^ M2)

            M0, M1, M2 = bi ,M0, M1

            return (M0 << 2) | (M1 << 1) | M2, ui0, ui1


        # PAD_BITS are derived from figures 121 and 122. It looks like they could be set arbitrarily
        if (data.size//8) % 2 == 0:
            # is even
            # L_PAD = 13
            PAD_BITS = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1])
        else:
            # L_PAD = 5
            PAD_BITS = np.array([0, 1, 0, 1, 1])

        M = 0b000

        encoded_PHR_PSDU = []
        for bi in data:
            # Split M into its bit values
            if self._phyFSKFECScheme:
                M, ui0, ui1 = M_iter_RSC(M, bi)
            else:
                M, ui0, ui1 = M_iter_NRNSC(M, bi)


            encoded_PHR_PSDU.append(ui1)
            encoded_PHR_PSDU.append(ui0)

        # Add tails bits and pad bits
        TAIL_BITS = RSC_TAIL_BITS[M] if self._phyFSKFECScheme else NRNSC_TAIL_BITS

        for bi in np.concatenate([TAIL_BITS, PAD_BITS]):
            if self._phyFSKFECScheme:
                M, ui0, ui1 = M_iter_RSC(M, bi)
            else:
                M, ui0, ui1 = M_iter_NRNSC(M, bi)

            encoded_PHR_PSDU.append(ui1)
            encoded_PHR_PSDU.append(ui0)

        return np.array(encoded_PHR_PSDU)

    def _interleaver(self, data):
        """
        Applies interleaver to the data

        Parameters
        ----------
        data : ndarray
            Input message (bitstream)

        Returns
        -------
        output : ndarray
        """
        output = np.zeros_like(data)

        # NOTE : Each permutation is applied on a pair of bits

        BLOCK_SIZE = 16

        k = np.arange(BLOCK_SIZE, dtype=int)
        t = (15 - 4 * np.mod(k, 4) - np.floor(k / 4)).astype(int)

        for i, block in enumerate(data.reshape(-1, BLOCK_SIZE * 2)):
            new_data = np.zeros_like(block)
            for ki, ti in zip(k, t):
                new_data[ti*2:ti*2 + 2] = block[ki*2:ki*2 + 2]
                output[i*BLOCK_SIZE*2:(i+1)*BLOCK_SIZE*2] = new_data

        return output

    def message_to_IQ(self, message, binary=False):
        """
        Encodes the given message with MR-FSK modulator

        Parameters
        ----------
        message : ndarray
            Message to encode (PSDU)
        binary : bool
            Specifies if the message is a bit array or a byte array.
            If the message is a byte array, it can be of type bytearray or a ndarray of integers

        Returns
        -------
        signal : ndarray
            output bitstream
        f : float
            signal frequency
        """
        message_bin = to_binary_array(message, binary)

        PHR_PSDU = np.concatenate(
            [self._PHR(message_bin.size // 8), message_bin])

        # Symbol_length is the number of bits coded for a single symbol. If FEC is disabled, there's one bit per symbol. If FEC is enabled, there's two bits per symbol

        if self._phyFSKFECEnabled:
            symbol_length = 2
            self._PHR_PSDU_encoded = self._FEC(PHR_PSDU)

            if self._phyFSKFECInterleavingRSC:
                self._PHR_PSDU_interleaved = self._interleaver(
                    self._PHR_PSDU_encoded)
                PHR_PSDU = self._PHR_PSDU_interleaved
            else:
                # Data is unchanged (encoding only)
                PHR_PSDU = self._PHR_PSDU_encoded
        else:
            # Do not change anything
            symbol_length = 1

        # Apply data whitening (or not)
        if self._phyFSKScramblePSDU:
            PSDU_start = PHR_LENGTH * symbol_length * 8

            self._PHR_PSDU_scrambled = PHR_PSDU.copy()

            self._PHR_PSDU_scrambled[PSDU_start:] = operations.scrambler(
                self._PHR_PSDU_scrambled[PSDU_start:])
            PHR_PSDU = self._PHR_PSDU_scrambled
        # Generate output signal
        signal = np.concatenate([
            self._SHR(),
            PHR_PSDU
        ])
        f = 0

        return signal, f

    def mode_switch_to_IQ(self, modeSwitchParameterEntry, new_mode_fec):
        """

        new_mode_fec : bool
            Signal that the packet following

        modeSwitchParameterEntry : int
            Mode switch operation (0-3)
        """

        if isinstance(modeSwitchParameterEntry, int):
            if not (0 <= modeSwitchParameterEntry <= 3):
                raise ValueError(
                    "Invalid modeSwitchParameterEntry value. It should be between 0 and 3")
        else:
            raise TypeError(
                "Invalid modeSwitchParameterEntry type. It should be int")
