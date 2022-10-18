from ..tools.errors import UnsupportedError
from ..mr_ofdm.mr_ofdm_modulator import Mr_ofdm_modulator


def test_parameters():
    """
    Create instances of modulators with all possible combinations and test if the
    message is encoded

    This function does not test whether or not the output signal is valid
    """
    def is_unsuported(fb, rm, sm):
        output = False
        return output


    for MCS in range(8):
        for OFDM_Option in range(1, 5):
            for phyOFDMInterleaving in range(2):
                for scrambler in range(4):
                    # Instanciate modulator
                    modulator = Mr_ofdm_modulator(MCS=MCS, OFDM_Option=OFDM_Option, phyOFDMInterleaving=phyOFDMInterleaving, scrambler=scrambler)
                    modulator.messageToIQ(b'test_message')