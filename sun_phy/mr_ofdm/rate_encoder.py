import math
import numpy as np


class Rate_one_half():
    def __init__(self, seed=0):
        """
        1/2 rate encoder
        See 802.15.4g figure 134

        Parameters
        ----------
        seed : int
            Initial value for the flip-flops
        """
        self._value = (seed & 0x3F)

    def reset(self, seed=0):
        """
        Resets the flip-flops to the given seed value (0 by default)
        """
        self._value = seed

    def single(self, input_bit):
        """
        Returns corresponding value for the given bit 
        Parameters
        ----------
        input_bit : int

        Returns
        -------
        A : int 
            Output data A
        B : int
            Output data B
        """
        input_bit = 1 if input_bit > 0 else 0
        A = input_bit ^ ((self._value & 0x10) >> 4) ^ ((self._value & 0x08) >> 3) ^ (
            (self._value & 0x02) >> 1) ^ (self._value & 0x01)
        B = input_bit ^ ((self._value & 0x20) >> 5) ^ ((self._value & 0x10) >> 4) ^ (
            (self._value & 0x08) >> 3) ^ (self._value & 0x01)
        self._value = (self._value >> 1) | (0x20 if input_bit else 0)
        return A, B

    def sequence(self, input_bits):
        """
        Returns a sequence corresponding to the input sequence
        Paramters
        ---------
        input_bits : ndarray
            Input list of bits (ints)
        Returns
        -------
        A : ndarray
            A output bits
        B : ndarray
            B output bits
        out : ndarray
            Concatenated output (A and B interlaced)
        """
        A = B = np.zeros_like(input_bits)
        out = np.zeros(input_bits.size * 2)

        for i, bit in enumerate(input_bits):
            a, b = self.single(bit)
            A[i] = a
            B[i] = b
            out[2*i], out[2*i+1] = a, b

        return A, B, out


class Rate_three_quarter():
    def __init__(self, seed=0):
        """
        3/4 rate encoder, similar to 1/2 encoder but with ommited bits
        Parameters
        ----------
        seed : int
            Initial value for the flip flops
        """
        self._rate_one_half = Rate_one_half(seed=seed)

    def sequence(self, input_bits):
        """
        Outputs A and B sequences from the input sequence
        Parameters
        ----------
        input_bits : ndarray
            input sequence

        Returns
        -------
        out : ndarray
            Output sequence
        """
        if input_bits.size % 3 > 0:
            raise ValueError("input sequence size must be a multiple of 3")

        out = np.zeros(int(input_bits.size * 4 / 3))
        for i, seq3 in enumerate(input_bits.reshape(-1, 3)):
            A, B, _ = self._rate_one_half.sequence(seq3)
            out[4*i] = A[0]
            out[4*i + 1] = B[0]
            out[4*i + 2] = A[1]
            out[4*i + 3] = B[2]

        return out
