from sun_phy import Mr_ofdm_modulator
import numpy as np

tables_path = "sun_phy/tests/"

MODULATOR_SETTINGS = {
    "MCS": 3,
    "OFDM_Option": 2,
    "phyOFDMInterleaving": False
}

message = np.genfromtxt(tables_path + "L.1.csv", delimiter=',').astype(int)

message_binary = []
for m in message:
    message_binary += [int(x) for x in np.binary_repr(m, 8)[::-1]]


def test_header():
    """
    Tests if the modulator generates a valid header according to tables L.2, L.3, L.4
    """

    modulator = Mr_ofdm_modulator(**MODULATOR_SETTINGS)

    # Run PHR modulator
    modulator._PHR(message.size * 8)

    # Read the theorethical values from the norm
    header_th = np.genfromtxt(tables_path + "L.2.csv",
                              delimiter=',').astype(int)
    header_encoded_th = np.genfromtxt(
        tables_path + "L.3.csv", delimiter=',').astype(int)
    header_interleaved_th = np.genfromtxt(
        tables_path + "L.4.csv", delimiter=',').astype(int)

    # Assert equality
    assert np.array_equal(modulator._PHY_header,
                          header_th), "Header doesn't match with table L.2"
    assert np.array_equal(modulator._PHY_header_encoded,
                          header_encoded_th), "Header doesn't match with table L.3"
    assert np.array_equal(modulator._PHY_header_interleaved,
                          header_interleaved_th), "Header doesn't match with table L.4"


def test_payload():
    """
    Tests if the modulator generates a valid payload according to tables L.7, L.8, L.9
    """
    # Load the message
    message = np.genfromtxt(tables_path + "L.1.csv", delimiter=',').astype(int)

    modulator = Mr_ofdm_modulator(**MODULATOR_SETTINGS)

    # Run complete modulator
    modulator.messageToIQ(message_binary)

    # Read the theorethical values from the norm
    payload_scrambled_th = np.genfromtxt(
        tables_path + "L.7.csv", delimiter=',').astype(int)
    paylaod_encoded_th = np.genfromtxt(
        tables_path + "L.8.csv", delimiter=',').astype(int)
    payload_interleaved_th = np.genfromtxt(
        tables_path + "L.9.csv", delimiter=',').astype(int)

    # Assert equality
    assert np.array_equal(modulator._payload_scrambled[:48], payload_scrambled_th[:48]) and np.array_equal(
        modulator._payload_scrambled[-48:], payload_scrambled_th[-48:]), "Header doesn't match with table L.7"
    assert np.array_equal(modulator._payload_encoded[:48], paylaod_encoded_th[:48]) and np.array_equal(
        modulator._payload_encoded[-48:], paylaod_encoded_th[-48:]), "Header doesn't match with table L.8"
    assert np.array_equal(modulator._payload_interleaved[:48], payload_interleaved_th[:48]) and np.array_equal(
        modulator._payload_interleaved[-48:], payload_interleaved_th[-48:]), "Header doesn't match with table L.9"


def test_time_domain():
    """
    Tests if the time domain signal is valid (STF, then LTF, then the complete signal)
    """
    # Instanciate modulator
    modulator = Mr_ofdm_modulator(**MODULATOR_SETTINGS)

    ERROR_THRESHOLD = 1e-6

    # Create STF
    STF_I, STF_Q = modulator._STF()
    STF = STF_I + STF_Q * 1j
    # Create LTF
    LTF_I, LTF_Q = modulator._LTF()
    LTF = LTF_I + LTF_Q * 1j
    # Create complete signal
    I, Q, _ = modulator.messageToIQ(message_binary)
    IQ = I + Q * 1j

    # Load theoretical data
    data_th = np.genfromtxt("sun_phy/tests/L.14.csv",
                            delimiter=',', dtype=float)

    data_th = data_th[:, 0] + data_th[:, 1] * 1j

    # Check STF
    STF_th = data_th[:STF.size]
    STF_error = np.var(STF_th - STF)
    assert STF_error < ERROR_THRESHOLD, f"STF error is outside bounds : {STF_error}"

    # Check LTF
    LTF_th = data_th[STF.size:STF.size + LTF.size]
    LTF_error = np.var(LTF_th - LTF)
    assert LTF_error < ERROR_THRESHOLD, f"LTF error is outside bounds : {LTF_error}"

    # Check complete signal
    IQ_error = np.var(data_th - IQ)
    assert IQ_error < ERROR_THRESHOLD, f"signal error is outside bounds : {IQ_error}"

