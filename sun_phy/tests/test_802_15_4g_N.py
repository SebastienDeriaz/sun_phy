from ..mr_o_qpsk.mr_o_qpsk_modulator import Mr_o_qpsk_modulator, Frequency_band
from .tests_tools import compare_arrays
import numpy as np
from os.path import join

tables_path = "sun_phy/tests/tables"

MODULATOR_SETTINGS = {
    "frequency_band" : Frequency_band.Band_470MHz,
    "rate_mode" : 0,
    "spreading_mode" : 0
}

message = np.genfromtxt(join(tables_path, "N.1.csv"), delimiter=',').astype(int)

message_binary = []
for m in message:
    message_binary += [int(x) for x in np.binary_repr(m, 8)[::-1]]
message_binary = np.array(message_binary)

def test_SHR():
    """
    Tests generation of the SHR
    """

    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)

    SHR = modulator._SHR()
    SHR_th = np.genfromtxt(join(tables_path, 'N.1_SHR.csv'), delimiter=',').astype(int)

    compare_arrays(SHR, SHR_th)

def test_SHR_BDE():
    """
    Tests SHR after BDE encoding
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)
    
    SHR_BDE = modulator._shr_bde
    SHR_BDE_th = np.genfromtxt(join(tables_path, 'N.1_SHR_BDE.csv'), delimiter=',').astype(int)

    compare_arrays(SHR_BDE, SHR_BDE_th)

def test_cSHR():
    """
    Tests generation of the chip sequence cSHR
    """

    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)
    
    cSHR = modulator._cSHR
    cSHR_th = np.genfromtxt(join(tables_path, 'N.2.csv'), delimiter=',').astype(int)

    compare_arrays(cSHR, cSHR_th)

def test_PHR():
    """
    Tests generation of the PHR
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)

    PHR = modulator._PHR(message_binary.size // 8)
    PHR_th = np.genfromtxt(join(tables_path, 'N.3.csv'), delimiter=',').astype(int)

    compare_arrays(PHR, PHR_th)

def test_PHR_encoded():
    """
    Tests FEC encoding of the PHR
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    PHR_encoded = modulator._phr_encoded
    PHR_encoded_th = np.genfromtxt(join(tables_path, 'N.4_PHR_encoded.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_encoded, PHR_encoded_th)

def test_PHR_interleaved():
    """
    Tests interleaving of the PHR
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    PHR_interleaved = modulator._phr_interleaved
    PHR_interleaved_th = np.genfromtxt(join(tables_path, 'N.4_PHR_interleaved.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_interleaved, PHR_interleaved_th)

def test_PHR_BDE():
    """
    Tests BDE encoding of PHR
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    PHR_BDE = modulator._phr_bde
    PHR_BDE_th = np.genfromtxt(join(tables_path, 'N.4_PHR_BDE.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_BDE, PHR_BDE_th)

def test_cPHR():
    """
    Test generation of the cPHR (DSSS encoding of PHR_BDE)
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    cPHR = modulator._cPHR
    cPHR_th = np.genfromtxt(join(tables_path, 'N.4.csv'), delimiter=',').astype(int)

    compare_arrays(cPHR, cPHR_th)

def test_PSDU_encoded():
    """
    Test encoded PSDU (pad + encoding)
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    PSDU_encoded = modulator._PSDU_encoded
    PSDU_encoded_th = np.genfromtxt(join(tables_path, 'N.5_PSDU_encoded.csv'), delimiter=',').astype(int)

    compare_arrays(PSDU_encoded, PSDU_encoded_th)

def test_PSDU_interleaved():
    """
    Tests interleaving of PSDU
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    PHR_interleaved = modulator._PSDU_interleaved
    PHR_interleaved_th = np.genfromtxt(join(tables_path, 'N.5_PSDU_interleaved.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_interleaved, PHR_interleaved_th)

def test_PSDU_BDE():
    """
    Tests BDE encoding of PSDU
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    PHR_BDE = modulator._PSDU_BDE
    PHR_BDE_th = np.genfromtxt(join(tables_path, 'N.5_PSDU_BDE.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_BDE, PHR_BDE_th)

def test_cPSDU():
    """
    Tests complete generation of cPSDU
    """
    modulator = Mr_o_qpsk_modulator(**MODULATOR_SETTINGS)
    modulator.bitsToIQ(message_binary)

    cPSDU = modulator._cPSDU
    cPSDU_th = np.genfromtxt(join(tables_path, 'N.5.csv'), delimiter=',').astype(int)

    compare_arrays(cPSDU, cPSDU_th)