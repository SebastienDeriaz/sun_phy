from ..tools.errors import UnsupportedError
from ..mr_o_qpsk.mr_o_qpsk_modulator import Frequency_band, Mr_o_qpsk_modulator, SpreadingMode


def test_parameters():
    """
    Create instances of modulators with all possible combinations and test if the
    message is encoded

    This function does not test whether or not the output signal is valid
    """
    def is_unsuported(fb, rm, sm):
        output = False
        if sm == SpreadingMode.MDSSS.value:
             if fb in [Frequency_band.Band_470MHz, Frequency_band.Band_868MHz, Frequency_band.Band_920MHz, Frequency_band.Band_950MHz]:
                output = True
        
        return output


    for fb in list(Frequency_band):
        for sm in range(2):
            for rm in range(4):
                if is_unsuported(fb, rm, sm):
                    # Catch Value Error
                    try:
                        modulator = Mr_o_qpsk_modulator(fb, rm, sm)
                    except UnsupportedError as e:
                        # Great ! we catched it
                        pass
                else:
                    modulator = Mr_o_qpsk_modulator(fb, rm, sm)
                    modulator.bytesToIQ(b'test_message')