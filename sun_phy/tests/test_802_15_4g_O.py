from ..mr_fsk.mr_fsk_modulator import Modulation, Mr_fsk_modulator
import numpy as np
from .tests_tools import compare_arrays
from os.path import join

tables_path = "sun_phy/tests/tables"

message = np.genfromtxt(join(tables_path, "L.1.csv"), delimiter=',').astype(int)

PSDU = np.array([int(x) for x in '01000000000000000101011001011101001010011111101000101000'])




MODULATOR_PARAMETERS = {
    # Example 1
    1 : {
        "phyMRFSKSFD" : 0,
        "phyFSKPreambleLength" : 4,
        "FCS_length" : 4,
        "modulation" : "2FSK",
        "phyFSKFECEnabled" : False,
        "phyFSKFECScheme" : 0, # unused
        "phyFSKFECInterleavingRSC" : False, # unused
        "phyFSKScramblePSDU" : False
    },
    # Example 2
    2 : {
        "phyMRFSKSFD" : 0,
        "phyFSKPreambleLength" : 4,
        "FCS_length" : 4,
        "modulation" : "2FSK",
        "phyFSKFECEnabled" : False,
        "phyFSKFECScheme" : 0, # unused
        "phyFSKFECInterleavingRSC" : False, # unused
        "phyFSKScramblePSDU" : True
        
    },
    # Example 3
    3 : {
        "phyMRFSKSFD" : 0,
        "phyFSKPreambleLength" : 4,
        "FCS_length" : 4,
        "modulation" : "2FSK",
        "phyFSKFECEnabled" : True,
        "phyFSKFECScheme" : 1,
        "phyFSKScramblePSDU" : True,
        "phyFSKFECInterleavingRSC" : True,
    },
    # Example 4
    4 : {
        "phyMRFSKSFD" : 1,
        "phyFSKPreambleLength" : 4,
        "FCS_length" : 4,
        "modulation" : "2FSK",
        "phyFSKFECEnabled" : True,
        "phyFSKFECScheme" : 0,
        "phyFSKScramblePSDU" : False,
        "phyFSKFECInterleavingRSC" : True,
    },
    # Example 5
    5 : {
        "phyMRFSKSFD" : 1,
        "phyFSKPreambleLength" : 4,
        "FCS_length" : 4,
        "modulation" : "4FSK",
        "phyFSKFECEnabled" : False,
        "phyFSKFECScheme" : 0, # unused
        "phyFSKScramblePSDU" : False,
        "phyFSKFECInterleavingRSC" : True,
    },
    # Example 6
    6 : {
        "phyMRFSKSFD" : 0,
        "phyFSKPreambleLength" : 4,
        "FCS_length" : 4,
        "modulation" : "4FSK",
        "phyFSKFECEnabled" : True,
        "phyFSKFECScheme" : 1, # unused
        "phyFSKScramblePSDU" : True,
        "phyFSKFECInterleavingRSC" : True,
    },

}



def test_Example1_O_2_2():
    """
    Test the generation of the SHR and compares it to O.2.2
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[1])

    SHR = modulator._SHR()

    SHR_th = np.genfromtxt(join(tables_path, 'O.2.2.csv'), delimiter=',').astype(int)

    compare_arrays(SHR, SHR_th)

def test_Example1_O_2_3():
    """
    Test the generation of the PHR
    """

    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[1])

    PHR_th = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    PHR = modulator._PHR(PSDU.size // 8)

    compare_arrays(PHR, PHR_th)

def test_Example1_O_2_4():
    """
    Test the generation of the PPDU (Concatenation of SHR, PHR and PSDU)
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[1])

    PPDU_th = np.genfromtxt(join(tables_path, 'O.2.4.csv'), delimiter=',').astype(int)
    
    PPDU, _ = modulator.message_to_bitstream(PSDU, binary=True)

    compare_arrays(PPDU, PPDU_th)

def test_Example2_O_3_2():
    """
    Compare the generation of the SHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[2])

    SHR = modulator._SHR()

    SHR_th = np.genfromtxt(join(tables_path, 'O.3.2.csv'), delimiter=',').astype(int)

    compare_arrays(SHR, SHR_th)

def test_Example2_O_3_3():
    """
    Compare the generation of the PHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[2])

    PHR = modulator._PHR(PSDU.size // 8)

    PHR_th = np.genfromtxt(join(tables_path, 'O.3.3.csv'), delimiter=',').astype(int)

    compare_arrays(PHR, PHR_th)

def test_Example2_O_3_5():
    """
    Compare the generation of the PPDU
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[2])

    PPDU, _ = modulator.message_to_bitstream(PSDU, binary=True)

    PPDU_th = np.genfromtxt(join(tables_path, 'O.3.5.csv'), delimiter=',').astype(int)

    compare_arrays(PPDU, PPDU_th)

def test_Example3_O_4_2():
    """
    Compare the generation of the SHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[3])

    SHR = modulator._SHR()
    SHR_th = np.genfromtxt(join(tables_path, 'O.4.2.csv'), delimiter=',').astype(int)

    compare_arrays(SHR, SHR_th)

def test_Example3_O_4_3():
    """
    Compare generation of the PHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[3])

    PHR = modulator._PHR(PSDU.size // 8)
    PHR_th = np.genfromtxt(join(tables_path, 'O.4.3.csv'), delimiter=',').astype(int)

    compare_arrays(PHR, PHR_th)

def test_Example3_O_4_5():
    """
    Compare encoded PHR + PSDU + Tail bits + pad bits
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[3])

    modulator.message_to_bitstream(PSDU, binary=True)

    PHR_PSDU_T_P = modulator._PHR_PSDU_encoded

    PHR_PSDU_T_P_th = np.genfromtxt(join(tables_path, 'O.4.5.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_PSDU_T_P, PHR_PSDU_T_P_th)

def test_Example3_O_4_6():
    """
    Compare the interleaving
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[3])

    modulator.message_to_bitstream(PSDU, binary=True)

    interleaved = modulator._PHR_PSDU_interleaved

    interleaved_th = np.genfromtxt(join(tables_path, 'O.4.6.csv'), delimiter=',').astype(int)
    
    compare_arrays(interleaved, interleaved_th)

def test_Example3_O_4_7():
    """
    Compare data whitening
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[3])

    modulator.message_to_bitstream(PSDU, binary=True)

    interleaved = modulator._PHR_PSDU_scrambled

    interleaved_th = np.genfromtxt(join(tables_path, 'O.4.7.csv'), delimiter=',').astype(int)
    
    compare_arrays(interleaved, interleaved_th)

def test_Example3_O_4_8():
    """
    Compare the generation of the PPDU
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[3])

    PPDU, _ = modulator.message_to_bitstream(PSDU, binary=True)
    PPDU_th = np.genfromtxt(join(tables_path, 'O.4.8.csv'), delimiter=',').astype(int)

    compare_arrays(PPDU, PPDU_th)

def test_Example4_O_5_1():
    """
    Compare generation of SHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[4])

    SHR = modulator._SHR()
    SHR_th = np.genfromtxt(join(tables_path, 'O.5.1.csv'), delimiter=',').astype(int)

    compare_arrays(SHR, SHR_th)

def test_Example4_O_5_2():
    """
    Compare generation of the PHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[4])

    PHR = modulator._PHR(PSDU.size // 8)
    PHR_th = np.genfromtxt(join(tables_path, 'O.5.2.csv'), delimiter=',').astype(int)

    compare_arrays(PHR, PHR_th)

def test_Example4_O_5_4():
    """
    Compare encoded PHR + PSDU + Tail bits + pad bits
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[4])

    modulator.message_to_bitstream(PSDU, binary=True)

    PHR_PSDU_T_P = modulator._PHR_PSDU_encoded

    PHR_PSDU_T_P_th = np.genfromtxt(join(tables_path, 'O.5.4.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_PSDU_T_P, PHR_PSDU_T_P_th)

def test_Example4_O_5_5():
    """
    Compare the interleaving
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[4])

    modulator.message_to_bitstream(PSDU, binary=True)

    interleaved = modulator._PHR_PSDU_interleaved

    interleaved_th = np.genfromtxt(join(tables_path, 'O.5.5.csv'), delimiter=',').astype(int)
    
    compare_arrays(interleaved, interleaved_th)

def test_Example4_O_5_6():
    """
    Compare the generation of the PPDU
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[4])

    PPDU, _ = modulator.message_to_bitstream(PSDU, binary=True)
    PPDU_th = np.genfromtxt(join(tables_path, 'O.5.6.csv'), delimiter=',').astype(int)

    compare_arrays(PPDU, PPDU_th)

def test_Example5_O_6_1():
    """
    Compare generation of SHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[5])

    SHR = modulator._SHR()
    SHR_th = np.genfromtxt(join(tables_path, 'O.6.1.csv'), delimiter=',').astype(int)

    compare_arrays(SHR, SHR_th)

def test_Example5_O_6_2():
    """
    Compare generation of the PHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[5])

    PHR = modulator._PHR(PSDU.size // 8)
    PHR_th = np.genfromtxt(join(tables_path, 'O.6.2.csv'), delimiter=',').astype(int)

    compare_arrays(PHR, PHR_th)

def test_Example5_O_6_4():
    """
    Compare the generation of the PPDU
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[5])

    PPDU, _ = modulator.message_to_bitstream(PSDU, binary=True)
    PPDU_th = np.genfromtxt(join(tables_path, 'O.6.4.csv'), delimiter=',').astype(int)

    compare_arrays(PPDU, PPDU_th)

def test_Example6_O_7_2():
    """
    Compare generation of SHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[6])

    SHR = modulator._SHR()
    SHR_th = np.genfromtxt(join(tables_path, 'O.7.2.csv'), delimiter=',').astype(int)

    compare_arrays(SHR, SHR_th)

def test_Example6_O_7_3():
    """
    Compare generation of the PHR
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[6])

    PHR = modulator._PHR(PSDU.size // 8)
    PHR_th = np.genfromtxt(join(tables_path, 'O.7.3.csv'), delimiter=',').astype(int)

    compare_arrays(PHR, PHR_th)

def test_Example6_O_7_5():
    """
    Compare encoded PHR + PSDU + Tail bits + pad bits
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[6])

    modulator.message_to_bitstream(PSDU, binary=True)

    PHR_PSDU_T_P = modulator._PHR_PSDU_encoded

    PHR_PSDU_T_P_th = np.genfromtxt(join(tables_path, 'O.7.5.csv'), delimiter=',').astype(int)

    compare_arrays(PHR_PSDU_T_P, PHR_PSDU_T_P_th)

def test_Example6_O_7_6():
    """
    Compare the interleaving
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[6])

    modulator.message_to_bitstream(PSDU, binary=True)

    interleaved = modulator._PHR_PSDU_interleaved

    interleaved_th = np.genfromtxt(join(tables_path, 'O.7.6.csv'), delimiter=',').astype(int)
    
    compare_arrays(interleaved, interleaved_th)

def test_Example6_O_7_7():
    """
    Compare data whitening
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[6])

    modulator.message_to_bitstream(PSDU, binary=True)

    interleaved = modulator._PHR_PSDU_scrambled

    interleaved_th = np.genfromtxt(join(tables_path, 'O.7.7.csv'), delimiter=',').astype(int)
    
    compare_arrays(interleaved, interleaved_th)


def test_Example6_O_7_8():
    """
    Compare the generation of the PPDU
    """
    modulator = Mr_fsk_modulator(**MODULATOR_PARAMETERS[6])

    PPDU, _ = modulator.message_to_bitstream(PSDU, binary=True)
    PPDU_th = np.genfromtxt(join(tables_path, 'O.7.8.csv'), delimiter=',').astype(int)

    compare_arrays(PPDU, PPDU_th)