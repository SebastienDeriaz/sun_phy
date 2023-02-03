from ..tools.errors import UnsupportedError
from ..mr_fsk.mr_fsk_modulator import Mr_fsk_modulator
from itertools import product
from random import choice
import numpy as np
import pytest

options_valid = {
    "symbolRate" : [10e3, 50e3, 1239821.3123],
    "FSKModulationIndex" : np.round(np.arange(0.25, 2.5, 0.05), 2).tolist(),
    "phyMRFSKSFD" : list(range(2)),
    "phyFSKPreambleLength" : [4],
    "modulation" : ["2FSK", "4FSK"],
    "phyFSKFECEnabled" : [False, True],
    "phyFSKFECScheme" : list(range(2)),
    "macFCSType" : list(range(2)),
    "phyFSKScramblePSDU" : [False, True],
    "phyFSKFECInterleavingRSC" : [False, True]
}

def gen_random(options : dict):
    return {key : choice(val) for key, val in options.items()}

N_valid_tests = 100

@pytest.mark.parametrize("parameters", [gen_random(options_valid) for _ in range(N_valid_tests)])
def test_parameters_valid(parameters):
    """
    Create instances of modulators with all possible combinations and test if the
    message is encoded

    This function does not test whether or not the output signal is valid
    """

    print(parameters)


    # Instanciate modulator
    modulator = Mr_fsk_modulator(**parameters)
    
    modulator.message_to_IQ(b'test_message', binary=False)