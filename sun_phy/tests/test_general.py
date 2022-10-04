from ..mr_ofdm.mr_ofdm_modulator import Mr_ofdm_modulator
from sun_phy import Ofdm_modulator


def test_instanciation():
    """
    Tests if it is possible to instantiate the MR-OFDM modulator 
    """
    mr_ofdm_modulator = Mr_ofdm_modulator()
    ofam_modulator = Ofdm_modulator()
