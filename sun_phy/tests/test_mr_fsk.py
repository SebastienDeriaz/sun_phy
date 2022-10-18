from ..tools.errors import UnsupportedError
from ..mr_fsk.mr_fsk_modulator import Mr_fsk_modulator


def test_parameters():
    """
    Create instances of modulators with all possible combinations and test if the
    message is encoded

    This function does not test whether or not the output signal is valid
    """
    def is_unsuported(fb, rm, sm):
        output = True        
        return output




    for phyMRFSKSFD in range(2):
        phyFSKPreambleLength = 4
        for modulation in ["2FSK", "4FSK"]:
            for phyFSKFECEnabled in [False, True]:
                for phyFSKFECScheme in range(2):
                    for macFCSType in range(2):
                        for phyFSKScramblePSDU in [False, True]:
                            for phyFSKFECInterleavingRSC in [False, True]:
                                # Instanciate modulator
                                modulator = Mr_fsk_modulator(
                                    phyMRFSKSFD=phyMRFSKSFD,
                                    phyFSKPreambleLength=phyFSKPreambleLength,
                                    modulation=modulation,
                                    phyFSKFECEnabled=phyFSKFECEnabled,
                                    phyFSKFECScheme=phyFSKFECScheme,
                                    macFCSType=macFCSType,
                                    phyFSKScramblePSDU=phyFSKScramblePSDU,
                                    phyFSKFECInterleavingRSC=phyFSKFECInterleavingRSC)
                                
                                modulator.message_to_bitstream(b'test_message', binary=False)