# Operations module
# Allows for multiple modulations to share common blocks
# Sébastien Deriaz
# 26.10.2022

import numpy as np
from .pn9 import Pn9


def scrambler(message, pn9_seed=511):
    """
    Applies scrambler (data whitening) to the message

    Parameters
    ----------
    message : ndaray
        Message's bitstream
    pn9_seed : int
        PN9 sequence intialization seed (511 by default)

    Returns
    -------
    output : ndarray
        Scrambled message
    """
    enc = Pn9(seed=pn9_seed)
    sequence = np.array(enc.nextN(message.size), dtype=np.uint8)
    output = np.bitwise_xor(sequence, message.astype(np.uint8))

    return output
