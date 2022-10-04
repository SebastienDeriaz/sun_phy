class Pn9():
    def __init__(self, seed=0x1FF):
        """
        PN9 Pseudo-random generator

        Parameters
        ----------
        seed : int
            initial value for all flip-flops
        """
        self._seed = seed
        self._value = seed

    def reset(self, seed=None):
        """
        Resets the flip-flops

        Parameters
        ----------
        seed : int
            Reset value, if not specified the initial value will be used (when the class is instanciated)
        """
        if seed is None:
            self._value = self._seed
        else:
            self._value = seed

    def next(self) -> int:
        """
        Returns the next value of the PN9 algorithm

        Returns
        -------
        nxt : int
            next value
        """
        self._value = ((self._value >> 1) & 0xFF) | (((self._value & 0x032) >> 0x05) ^ (self._value & 0x001)) << 8
        return (self._value & 0x100) >> 8

    def nextN(self, N) -> list:
        """
        Returns the N next values of the PN9 algorithm

        Returns
        -------
        nxt_lst : list
            List of next N values of the alogrithm
        """
        sequence = []
        for i in range(N):
            sequence.append(self.next())
        return sequence

    def get_current_value(self):
        """
        Gets the current flip-flops values as an integer

        Returns
        -------
        value : int
            Value of the flip-flops
        """
        return self._value




