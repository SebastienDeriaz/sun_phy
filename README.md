# sun_phy
 A Python implementation of 802.15.4g LR-WPANs SUN PHYs : MR-FSK, MR-OFDM and MR-O-QPSK


## Installation

``pip install sun-phy``

## Usage

### MR-FSK

```python

from sun_phy import Mr_fsk_modulator

modulator = Mr_fsk_modulator(
    # Set these parameters
    phyMRFSKSFD=0,
    phyFSKPreambleLength=4,
    modulation='2FSK',
    phyFSKFECEnabled=True,
    phyFSKFECScheme=0,
    macFCSType=0,
    phyFSKScramblePSDU=True,
    phyFSKFECInterleavingRSC=False)

# The message can also be sent as a numpy array of bytes or bits
I, Q, f = modulator.message_to_IQ(b'my message')

I, Q, f = modulator.mode_switch_to_IQ(
    # Set these parameters
    modeSwitchParameterEntry=0,
    new_mode_fec=0)
```

### MR-OFDM

```python

from sun_phy import Mr_ofdm_modulator

modulator = Mr_ofdm_modulator(
    # Set these parameters
    MCS=3,
    OFDM_Option=2,
    phyOFDMInterleaving=0,
    scrambler=0,
    verbose=False # Verbose True enables printing of debugging info
)

# Similarly to MR-FSK, the message can by a byte or bits array
I, Q, f = modulator.message_to_IQ(b'my message')
```

### MR-O-QPSK

```python

from sun_phy import Mr_o_qpsk_modulator, Frequency_band

modulator = Mr_o_qpsk_modulator(
    # Set these parameters
    frequency_band=Frequency_band.Band_470MHz,
    rate_mode=0,
    spreading_mode=0)

I, Q, f = modulator.message_to_IQ(b'my message')
```

