import numpy as np
from .ofdm_modulator import Ofdm_modulator
from sun_phy.tools.pn9 import Pn9
from .rate_encoder import Rate_one_half, Rate_three_quarter
from .fields import PHR, TAIL_BITS

from colorama import Fore


# FFT size as function of OFDM option
FFT_SIZE = {
    1: 128,
    2: 64,
    3: 32,
    4: 16}

# Shape of pilots for each
# Number of rows : pilots in a symbol
# Number of columns : amout of pilot sets (change of positions)
PILOTS_SHAPE = {
    1: (8, 13),
    2: (4, 7),
    3: (2, 7),
    4: (2, 4)}

# Position of pilots
# (See Table 153 page 88 of 802.15.4g)
# warning : The array is transposed (see .T at the end)
PILOTS_INDICES_OPTION_1 = np.array([
    [-38, -26, -14, -2, 10, 22, 34, 46],  # Set 1
    [-46, -34, -22, -10, 2, 14, 26, 38],  # Set 2
    [-42, -30, -18, -6, 6, 18, 30, 42],  # Set 3
    [-50, -38, -26, -14, -2, 10, 22, 50],  # Set 4
    [-46, -34, -22, -10, 2, 14, 34, 46],  # Set 5
    [-42, -30, -18, -6, 6, 18, 26, 38],  # Set 6
    [-50, -38, -26, -14, -2, 30, 42, 50],  # Set 7
    [-46, -34, -22, -10, 10, 22, 34, 46],  # Set 8
    [-42, -30, -18, -6, 2, 14, 26, 38],  # Set 9
    [-50, -38, -26,  6, 18, 30, 42, 50],  # Set 10
    [-46, -34, -14, -2, 10, 22, 34, 46],  # Set 11
    [-42, -30, -22, -10, 2, 14, 26, 38],  # Set 12
    [-50, -18, -6, 6, 18, 30, 42, 50]  # Set 13
]).T

# See table 154
PILOTS_INDICES_OPTION_2 = np.array([
    [-14, -2, 10, 22],  # Set 1
    [-22, -10, 2, 14],  # Set 2
    [-18, -6, 6, 18],  # Set 3
    [-26, -14, -2, 26],  # Set 4
    [-22, -10, 10, 22],  # Set 5
    [-18, -6, 2, 14],  # Set 6
    [-26,  6, 18, 26]  # Set 7
]).T

# See table 155
PILOTS_INDICES_OPTION_3 = np.array([
    [-7, 7],  # Set 1
    [-11, 3],  # Set 2
    [-3, 11],  # Set 3
    [-9, 5],  # Set 4
    [-5, 9],  # Set 5
    [-13, 1],  # Set 6
    [-1, 13]  # Set 7
]).T

# See table 156
PILOTS_INDICES_OPTION_4 = np.array([
    [-3, 5],  # Set 1
    [-7, 1],  # Set 2
    [-5, 3],  # Set 3
    [-1, 7]  # Set 4
]).T

# Pilots indices as a function of OFDM option
PILOTS_INDICES = {
    1: PILOTS_INDICES_OPTION_1,
    2: PILOTS_INDICES_OPTION_2,
    3: PILOTS_INDICES_OPTION_3,
    4: PILOTS_INDICES_OPTION_4
}

# See Table 140
STF_OPTION_1 = np.zeros(FFT_SIZE[1])
STF_OPTION_1[np.array([-48, -40, -32, -24, -16, -8, 8, 16,
                      24, 32, 40, 48])+STF_OPTION_1.size//2] = np.sqrt(104/12)

# See Table 141
STF_OPTION_2 = np.zeros(FFT_SIZE[2])
STF_OPTION_2[np.array([-24, -20, -16, 8, 20]) +
             STF_OPTION_2.size//2] = -np.sqrt(52/12)
STF_OPTION_2[np.array([-12, -8, -4, 4, 12, 16, 24]) +
             STF_OPTION_2.size//2] = np.sqrt(52/12)

# See Table 142
STF_OPTION_3 = np.zeros(FFT_SIZE[3])
STF_OPTION_3[np.array([-12, -8, -4, 4, 8, 12]) +
             STF_OPTION_3.size//2] = np.sqrt(26/6)

# See Table 143
STF_OPTION_4 = np.zeros(FFT_SIZE[4])
STF_OPTION_4[np.array([-6, -4, -2, 2, 4, 6]) +
             STF_OPTION_4.size//2] = np.sqrt(14/6)

# The square roots are due to Power Boosting (See 18.2.1.1.4)

STF = {
    1: STF_OPTION_1,
    2: STF_OPTION_2,
    3: STF_OPTION_3,
    4: STF_OPTION_4
}
# Modulation as a function of MCS (See table )
MODULATION_AND_CODING_SCHEME = {
    0: 'BPSK',
    1: 'BPSK',
    2: 'QPSK',
    3: 'QPSK',
    4: 'QPSK',
    5: 'QAM16',
    6: 'QAM16'
}

N_BPSC = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 4,
    6: 4
}

RATE = {
    0: "1/2",
    1: "1/2",
    2: "1/2",
    3: "1/2",
    4: "3/4",
    5: "1/2",
    6: "3/4"
}

# Modulation factor (See table 149)
K_MOD = {
    'BPSK': 1,
    'QPSK': 1/np.sqrt(2),
    'QAM16': 1/np.sqrt(10)
}

# Number of active tones (pilots + data) for each option
ACTIVE_TONES = {
    1: 104,
    2: 52,
    3: 26,
    4: 14
}

# Number of data tones (active tones - pilot tones)
DATA_TONES = {
    1: 96,
    2: 48,
    3: 24,
    4: 12
}

# Frequency spreading depending on MCS
FREQUENCY_SPREADING = {
    0: 4,
    1: 2,
    2: 2,
    3: 1,
    4: 1,
    5: 1,
    6: 1
}


# Valid MCS - OFDM option combinations (see table 148 page 80)
VALID_MCS_OFDM_COMBINATIONS = (
    (True, True, False, False),
    (True, True, True, False),
    (True, True, True, True),
    (True, True, True, True),
    (False, True, True, True),
    (False, True, True, True),
    (False, False, True, True),
)

LOWEST_MCS_VALUE = {
    1: 0,
    2: 0,
    3: 1,
    4: 2
}


# Spacing between FFT channels (See 18.2 page 70)
SUB_CARRIER_SPACING = 31250/3  # Hz

# Long training fields (LTF) subcarriers
# LTF1 -> 128 elements
LTF_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1,
                 1, 1, 0, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# LTF2 -> 64 elements
LTF_2 = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -
                 1, -1, 0, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 0, 0, 0, 0, 0])
# LTF3 -> 32 elements
LTF_3 = np.array([0, 0, 0, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1,
                 1, -1, 0, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 0, 0])
# LTF4 -> 16 elements
LTF_4 = np.array([0, 1, -1, 1, 1, -1, 1, 1, 0, -1, 1, 1, 1, -1, -1, -1])

LTF = {
    1: LTF_1,
    2: LTF_2,
    3: LTF_3,
    4: LTF_4
}

# Scrambling seed (See Table 158)
# bit 19 of PHR is MSB of key (S1)
# bit 20 of PHR is LSB of key (s0)
SCRAMBLING_SEED = {
    0: int('0b000010111', 2),
    1: int('0b000011100', 2),
    2: int('0b101110111', 2),
    3: int('0b101111100', 2)
}

_Color0 = Fore.CYAN
_ColorReset = Fore.RESET


class Mr_ofdm_modulator():
    def __init__(self, MCS=0, OFDM_Option=1, phyOFDMInterleaving=0, scrambler=0, verbose=False):
        """
        MR-OFDM Modulator

        Parameters
        ----------
        MCS : int
            Modulation coding scheme (0 - 7)
        OFDM_Option : int
            OFDM Option (1 - 4)
        phyOFDMInterleaving : int (0 or 1)
            Interleaving depth. 0 means an interleaving depth of one. 1 means an interleaving depth of the number of symbols equal to the frequency domain spreading factor
        scrambler : int
            Scrambler value (0-3). See 18.2.3.11
        verbose : bool
            if True, print additionnal information
        """
        # Check types and values
        if not isinstance(MCS, int):
            raise ValueError("MCS must be int")
        if not (0 <= MCS <= 7):
            raise ValueError(f"Invalid MCS value ({MCS} instead of [0 - 7])")
        if not isinstance(OFDM_Option, int):
            raise ValueError("OFDM_Option must be int")
        if not (1 <= OFDM_Option <= 4):
            raise ValueError(
                f"Invalid OFDM_Option ({OFDM_Option} instead of [1-4])")
        if not VALID_MCS_OFDM_COMBINATIONS[MCS][OFDM_Option-1]:
            raise ValueError(
                f"Invalid MCS ({MCS})- OFDM Option combination ({OFDM_Option})")
        if not isinstance(phyOFDMInterleaving, int):
            raise ValueError("Invalid phyOFDMInterleaving type")
        if not phyOFDMInterleaving in [0, 1]:
            raise ValueError("Invalid phyOFDMInterleaving value")
        if not isinstance(scrambler, int):
            raise ValueError("Invalid scrambler type. Must be int")
        if not (0 <= scrambler <= 3):
            raise ValueError("Invalid scrambler value (0 <= scrambler <= 3)")

        self._phyOFDMInterleaving = phyOFDMInterleaving
        self._scrambler_seed_index = scrambler
        self._OFDM_Option = OFDM_Option
        self._verbose = verbose
        self._MCS = MCS
        # Lowest possible MCS value for the current OFDM option (for sending PHR)
        self._lowest_MCS = LOWEST_MCS_VALUE[OFDM_Option]

        self._N_cbps = DATA_TONES[OFDM_Option]
        #self._N_dbps = DATA_TONES[OFDM_Option] * N_BPSC[self._MCS] # this is wrong
        self._rate = 1/2 if RATE[self._MCS] == "1/2" else 3/4
        self._N_dbps = int(self._N_cbps * self._rate)

        self._CP = 1/4

        self._print_verbose(_Color0 + 
            f"Instanciating a modulator with OFDM Option = {OFDM_Option} and MCS = {MCS}" + _ColorReset)

        

    def _STF(self):
        """
        Generates STF signal (as I and Q)
        The second half of the fourth symbol is inverted in the time domain :

          symbol 1   symbol 2   symbol 3   symbol 4
        |CP++++++++|CP++++++++|CP++++++++|CP++++----|

        +  : samples of the symbol (after IFFT)
        CP : cyclic prefix
        -  : negated samples

        Returns
        -------
        I : ndarray
            Real part of the signal
        Q : ndarray
            Imaginary part of the signal
        """
        self._print_verbose("Creating STF....")
        self._print_verbose(
            """
          symbol 1   symbol 2   symbol 3   symbol 4
        |CP++++++++|CP++++++++|CP++++++++|CP++++----|

        +  : samples of the symbol (after IFFT)
        CP : cyclic prefix
        -  : negated samples (end of fourth symbol)
        """)

        STF_CP = 1/4
        POWER_BOOSTING_FACTOR = 1.25  # See 18.2.1.1.4
        # OFDM modulator for the STF symbols specifically
        mod = Ofdm_modulator(N_FFT=FFT_SIZE[self._OFDM_Option], CP=STF_CP)

        # Time domain STF symbol with cyclic prefix
        STF_I, STF_Q, _ = mod.subcarriersToIQ(STF[self._OFDM_Option])
        STF_time_domain = (STF_I + STF_Q * 1j).squeeze()
        # Fourth symbol (inverted end)
        STF_time_domain_fourth = STF_time_domain * \
            np.block([np.ones(STF_time_domain.size // 5 * 3),
                     -np.ones(STF_time_domain.size // 5 * 2)])
        # Create the signal with the four symbols
        signal = np.block([STF_time_domain, STF_time_domain,
                          STF_time_domain, STF_time_domain_fourth])
        I, Q = signal.real * POWER_BOOSTING_FACTOR, signal.imag * POWER_BOOSTING_FACTOR

        self._print_verbose(f"    STF signal is {signal.size} elements")

        return I, Q

    def _LTF(self):
        """
        Generates Long Training Field (LTF) signal in the time domain

        The LTF contains a double-size cyclic prefix (CP) at the beginning and two LTF symbols.
        The CP is calculated on the second half of the first symbol (both symbols are identical anyway)
        The time domain signal looks something like this :

        |CP----|----|

        Returns
        -------
        I : ndarray
            Real part of the signal
        Q : ndarray
            Imaginary part of the signal
        """
        self._print_verbose("Generating LTF...")
        self._print_verbose("    LTF signal looks like : |CP----|----|")

        # OFDM modulator for the LTF excusively. No padding because we will add it manually
        mod = Ofdm_modulator(N_FFT=FFT_SIZE[self._OFDM_Option], CP=0)
        # LTF signal without cyclic prefix (CP)
        LTF_I, LTF_Q, _ = mod.subcarriersToIQ(LTF[self._OFDM_Option])
        LTF_signal = (LTF_I + 1j*LTF_Q).squeeze()
        # Create the cyclic prefix (second half of the first symbol)
        CP = LTF_signal[LTF_signal.size//2:]
        # Create the complete signal
        signal = np.block([CP, LTF_signal, LTF_signal])
        I, Q = signal.real, signal.imag

        self._print_verbose(f"    LTF signal is {signal.size} elements")

        return I, Q

    def _encoder(self, x):
        """
        Applies encoding to the given signal
        See 18.2.3.4

        Rate is 1/2 or 3/4 depending on MCS value

        Parameters
        ----------
        x : ndarray
            Input signal

        Returns
        -------
        x_encoded : ndarray
            Encoded signal
        """
        if RATE[self._MCS] == "1/2":
            # Rate 1/2
            encoder = Rate_one_half()
        else:
            # Rate 3/4
            encoder = Rate_three_quarter()

        _, _, x_encoded = encoder.sequence(x)

        return x_encoded

    def _interleaver(self, x, MCS):
        """
        Applies interleaver to the given signal
        See 18.2.3.5

        Parameters
        ----------
        x : ndarray
            Input signal
        MCS : int
            The current MCS value

        Returns
        -------
        x_interleaved : ndarray
            Interleaved signal
        """
        SF = FREQUENCY_SPREADING[MCS]
        N_FFT = FFT_SIZE[self._OFDM_Option]
        N_bpsc = N_BPSC[MCS]

        if self._phyOFDMInterleaving == 0:
            # interleaving depth of 1
            N_cbps = int(np.round(N_FFT * N_bpsc / SF * (3/4)))
        else:
            # interleaving depth of SF
            # See table
            N_cbps = int(np.round(N_FFT * N_bpsc * (3/4)))
        # 06.05.2022, testing here (instead of inside the else:)
        N_row = 12 // SF

        k = np.arange(N_cbps, dtype=int)
        i = ((N_cbps / N_row) * (np.mod(k, N_row)) +
             np.floor(k / N_row)).astype(int)

        s = np.max([1, N_bpsc/2])
        # We use i as an index here, so we store a copy of the previous one
        i_store = i.copy()
        i = np.arange(N_cbps, dtype=int)
        j = (s * np.floor(i / s) + np.mod(i + N_cbps -
             np.floor(N_row * i / N_cbps), s)).astype(int)

        ij = i_store[j]

        x_interleaved = np.zeros_like(x)
        for s in range(x.size // k.size):
            x_interleaved[s*k.size + ij] = x[s*k.size + k]

        return x_interleaved

    def _padding(self, message):
        """
        Adds padding to the given message. The number of padding bits is calculated according to 18.2.3.10

        Parameters
        ----------
        message : ndarray
            The message to pad
        length : int
            The size of the transmitted message

        Returns
        -------
        output : ndarray
            The padded message
        """
        N_TAIL_BITS = 6
        length = message.size // 8 # number of octets
        # See P.90
        N_SYM = np.ceil((8 * length + 6) / self._N_dbps)
        N_DATA = N_SYM * self._N_dbps
        N_PAD = int(N_DATA - (8 * length + 6))
        self._N_PAD = N_PAD
        self._print_verbose(f"Adding {N_PAD} padding bits and {N_TAIL_BITS} tail bits to the message")
        output = np.block([message, np.zeros(N_PAD), np.zeros(N_TAIL_BITS)])
        return output

    def _scrambler(self, message):
        """
        Applies scrambling to the message. The message is the PHR with pad and tail bits

        Parameters
        ----------
        message : ndarray
            The message to scrambler

        Returns
        -------
        output : ndarray
            Scrambled message
        """
        N_TAIL_BITS = 6
        enc = Pn9(seed=SCRAMBLING_SEED[self._scrambler_seed_index])
        sequence = np.array(enc.nextN(message.size), dtype=np.uint8)
        output = np.bitwise_xor(sequence, message.astype(np.uint8))

        # Reset the tail bits to 0
        output[-N_TAIL_BITS-self._N_PAD:-self._N_PAD] = 0
        return output

    def _PHR(self, message_length):
        """
        Generate a PHR
        
        Parameters
        ----------
        message_length : int
            Length of the message

        Returns
        -------
        I : ndarray
            Real part of the signal
        Q : ndarray
            Imaginary part of the signal
        mod_phy : Ofdm_modulator
            OFDM modulator used for the PHR
        """

        frame_length = message_length // 8
        self._print_verbose(Fore.LIGHTBLUE_EX + "Generating PHY header...")
        self._PHY_header = PHR(rate=self._MCS, length=frame_length,
                         scrambler=self._scrambler_seed_index, phyOFDMInterleaving=self._phyOFDMInterleaving).value()
        self._print_verbose(Fore.LIGHTBLUE_EX + f"header : {self._PHY_header.size} bits")
        # Encoding header
        self._PHY_header_encoded = self._encoder(self._PHY_header)
        self._print_verbose(Fore.LIGHTBLUE_EX + f"header after encoding : {self._PHY_header_encoded.size} bits")
        # Interleaving header
        self._PHY_header_interleaved = self._interleaver(
            self._PHY_header_encoded, self._lowest_MCS)
        self._print_verbose(Fore.LIGHTBLUE_EX + f"header after interleaving : {self._PHY_header_interleaved.size} bits")
        # Apply OFDM modulation
        # Sets the padding (one less on the right than on the left because of the DC tone)
        padding = (FFT_SIZE[self._OFDM_Option] -
                   ACTIVE_TONES[self._OFDM_Option]) // 2
        mod_phy = Ofdm_modulator(
            N_FFT=FFT_SIZE[self._OFDM_Option],
            modulation=MODULATION_AND_CODING_SCHEME[self._lowest_MCS],
            modulation_factor=K_MOD[MODULATION_AND_CODING_SCHEME[self._lowest_MCS]],
            frequency_spreading=FREQUENCY_SPREADING[self._lowest_MCS],
            padding_left=padding,
            padding_right=padding-1,
            pilots_indices=PILOTS_INDICES[self._OFDM_Option],
            pilots_values="pn9",
            CP=self._CP)

        PHR_I, PHR_Q, _ = mod_phy.messageToIQ(self._PHY_header_interleaved, pad=False)
        self._print_verbose(Fore.LIGHTBLUE_EX + f"header complex (I+jQ) signal is {PHR_I.size} samples" + Fore.RESET)

        return PHR_I, PHR_Q, mod_phy
    
    def _payload(self, message, mod_phy):
        """
        Generate Payload from message

        Parameters
        ----------
        message : ndarray
            Message to encode
        mod_phy : Ofdm_modulator
            OFDM modulator used for the PHR

        Returns
        -------
        I : ndarray
            Real part of the signal
        Q : ndarray
            Imaginary part of the signal
        """

        # Scrambler
        payload_pad = self._padding(message)
        self._payload_scrambled = self._scrambler(payload_pad)
        # Encoding header
        self._payload_encoded = self._encoder(self._payload_scrambled)
        # Interleaving header
        self._payload_interleaved = self._interleaver(
            self._payload_encoded, self._MCS)

        padding = (FFT_SIZE[self._OFDM_Option] -
                   ACTIVE_TONES[self._OFDM_Option]) // 2
        
        # Payload
        mod_payload = Ofdm_modulator(
            N_FFT=FFT_SIZE[self._OFDM_Option],
            modulation=MODULATION_AND_CODING_SCHEME[self._MCS],
            frequency_spreading=FREQUENCY_SPREADING[self._MCS],
            modulation_factor=K_MOD[MODULATION_AND_CODING_SCHEME[self._MCS]],
            padding_left=padding,
            padding_right=padding-1,
            pilots_indices=PILOTS_INDICES[self._OFDM_Option],
            pilots_values="pn9",
            CP=self._CP,
            initial_pilot_set = mod_phy.get_pilot_set_index(),
            initial_pn9_seed = mod_phy.get_pn9_value())

        PAYLOAD_I, PAYLOAD_Q, _ = mod_payload.messageToIQ(self._payload_interleaved, pad=True)

        return PAYLOAD_I, PAYLOAD_Q

    def messageToIQ(self, message):
        """
        Encodes the given message with MR-OFDM modulator

        Parameters
        ----------
        message : ndarray
            Message to encode

        Returns
        -------
        I : ndarray
            Real part of the signal
        Q : ndarray
            Imaginary part of the signal
        f : float
            I and Q signals frequency
        """
        # Convert list to numpy array if necessary
        if isinstance(message, list):
            message = np.asarray(message)
        # Generate STF
        STF_I, STF_Q = self._STF()
        # Generate LTF
        LTF_I, LTF_Q = self._LTF()

        # Generate header
        PHR_I, PHR_Q, mod_phy = self._PHR(message.size)

        # Generate Payload
        PAYLOAD_I, PAYLOAD_Q = self._payload(message, mod_phy)
        
        I = np.block([STF_I, LTF_I, PHR_I, PAYLOAD_I])
        Q = np.block([STF_Q, LTF_Q, PHR_Q, PAYLOAD_Q])

        # I and Q signals frequency
        # We know a symbol (with cyclic prefix) is 120us
        # Therefore we can determine the period with 120us / (FFT size * (1+CP))
        # And f = (FFT size * (1+CP)) / 120us
        SYMBOL_DURATION = 120e-6 # 120us
        f = FFT_SIZE[self._OFDM_Option] * (1+self._CP) / SYMBOL_DURATION

        return I, Q, f

    def _print_verbose(self, message: str):
        """
        Prints additionnal information if the verbose flag is True
        """
        if(self._verbose):
            print(message)
