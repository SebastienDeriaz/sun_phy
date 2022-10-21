from ..tools.errors import UnsupportedError
from ..mr_ofdm.mr_ofdm_modulator import Mr_ofdm_modulator


def test_parameters():
    """
    Create instances of modulators with all possible combinations and test if the
    message is encoded

    This function does not test whether or not the output signal is valid
    """
    def is_unsuported(MCS, OFDM_Option, phyOFDMInterleaving, scrambler):
        return (OFDM_Option == 1 and MCS not in [0, 1, 2, 3] ) or \
            (OFDM_Option == 2 and MCS not in [0, 1, 2, 3, 4, 5]) or \
            (OFDM_Option == 3 and MCS not in [1, 2, 3, 4, 5, 6]) or \
            (OFDM_Option == 4 and MCS not in [2, 3, 4, 5, 6])
            


    for MCS in range(7):
        for OFDM_Option in range(1, 5):
            for scrambler in range(4):
                for phyOFDMInterleaving in range(2):
                    # Instanciate modulator
                    if is_unsuported(MCS, OFDM_Option, phyOFDMInterleaving, scrambler):
                        # Expect an UnsupportedError
                        try:
                            modulator = Mr_ofdm_modulator(MCS=MCS, OFDM_Option=OFDM_Option, phyOFDMInterleaving=phyOFDMInterleaving, scrambler=scrambler)
                        except UnsupportedError as e:
                            # got it !
                            continue
                        else:
                            raise ValueError("Unsupported parameters combination wasn't raised")
                    else:
                        modulator = Mr_ofdm_modulator(MCS=MCS, OFDM_Option=OFDM_Option, phyOFDMInterleaving=phyOFDMInterleaving, scrambler=scrambler)
                    
                    modulator.message_to_IQ(b'test_message', binary=False)