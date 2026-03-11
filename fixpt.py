"""
Fixed-point arithmetic realization. Support for different saturation and quantization policies.

The main feature of this realization is that intermedia calculation are performed
in fixed points with arbitrary (automatically calculated) width and fraction bits number.
Reduction to fixed point with specified width and number of fracbits is always explicit operation.

Features
--------
* Automatic type expansion (width and number of fracbits) during arithmetic operation.
* Operation of reduction to specific fixed point type are explicit.
* Different saturation and quantization policies.

Example
-------
    # create fixed point type by specifying width, number of fractional bits, quantization and saturation policies.
    ftype = FixedPointType(16, 8, QuantPolicy.TruncateToZero, SatPolicy.Saturation)

    # instal values
    a = FixedPoint(0.5, ftype)
    b = FixedPoint(0.25, ftype)
    c = FixedPoint(0.0, ftype)
    # alternative version
    c = ftype()

    # atiphmetic
    tmp = a + b # FixedPointBase with 8 fracbits
    tmp *= b    # FixedPointBase with 16 fracbits

    # reduce result to ftype fixed-point (apply quantization and saturation policy)
    c.assign(tmp)
    # alternative version with object creation
    c = FixedPoint(tmp, ftype)
    c = ftype(tmp)

"""

import math

#
# quantization methods
#


class QuantPolicy:
    """Quantization policies namespace."""

    class QuantBase:
        """Base class for all quantization policies. Do not use explicitly.

        Attributes
        ----------
        fracbits: int
            Number of fractional bits.

        """

        def __init__(self, fracbits):
            """Create quantization policy with given number of fractional bits.

            Parameters
            ----------
            fracbits: int
                Number of fractional bits.
            """
            self.fracbits = fracbits

        @property
        def name(self):
            """Return policy name."""
            return type(self).__name__

        def __call__(self, value):
            """Perform quantization (truncation) of given number according to quantization policy.

            Implement this function in subclass to create new quantozation policy.

            Parameters
            ----------
            value: float or FixedPointBase
                Value to truncate.

            Return
            ------
            int
                Int representation of fixed point.
            """
            raise NotImplementedError

    class Truncate(QuantBase):
        """Quantization by simple truncation: exact value is replaced by nearest smaller allowed value."""

        def __call__(self, value):
            if isinstance(value, FixedPointBase):
                rshift = value.fracbits - self.fracbits
                if rshift >= 0:
                    return value._int >> rshift
                else:
                    return value._int << (-rshift)
            elif isinstance(value, float):
                return math.floor(value * 2**self.fracbits)
            else:
                raise TypeError(f"FixedPointBase or float is expected, {type(value)} is received.")

    class TruncateToZero(QuantBase):
        """Quantization by symmetric truncation: exact value is replaced by nearest allowed value which is closer to zero."""

        def __call__(self, value):
            """Truncate value to the nearest allowed value which is closer zero."""
            if isinstance(value, FixedPointBase):
                rshift = value.fracbits - self.fracbits
                if rshift > 0:
                    if value._int >= 0:
                        return value._int >> rshift
                    else:
                        return (value._int >> rshift) + 1
                else:
                    return value._int << (-rshift)
            elif isinstance(value, float):
                return int(value * 2**self.fracbits)
            else:
                raise TypeError(f"FixedPointBase or float is expected, {type(value)} is received.")

    class Round(QuantBase):
        """Quantization by symmetric truncation: exact value is replaced by nearest allowed value."""

        def __call__(self, value):
            if isinstance(value, FixedPointBase):
                rshift = value.fracbits - self.fracbits
                if rshift > 0:
                    return ((value._int >> (rshift-1)) + 1) >> 1
                else:
                    return value._int << (-rshift)
            elif isinstance(value, float):
                return round(value * 2**self.fracbits)
            else:
                raise TypeError(f"FixedPointBase or float is expected, {type(value)} is received.")

    class Exception(QuantBase):  # noqa A001
        """Raise exception if quantization is necessary (precision is lost)."""

        def __call__(self, value):
            if isinstance(value, FixedPointBase):
                lshift = self.fracbits - value.fracbits
                if lshift >= 0:
                    return value._int << lshift
                else:
                    rshift = -lshift
                    mask = (1 << rshift) - 1
                    if not (value._int & mask):
                        return value._int >> rshift
                    else:
                        raise ValueError("QuantPolicy.Exception: unable to assign without rounding.")
            elif isinstance(value, float):
                int_value = value * 2**self.fracbits
                if int_value.is_integer():
                    return int(int_value)
                else:
                    raise ValueError(f"QuantPolicy.Exception: {value} can not be exactly represented as fixed point with {self.fracbits} fracbits")
            else:
                raise TypeError(f"FixedPointBase or float is expected, {type(value)} is received.")

#
# saturation methods
#


class SatPolicy:
    """Saturation policy namespace."""

    class SatBase:
        """Saturation policy base class. Do not use explicitly.

        Attributes
        ----------
        min_value: int
           Lower saturation limit. Minimal allowed value in int representation.
        max_value: int
           Higher saturation limit. Maximal allowed value in int representation.

        """

        def __init__(self, bits: int):
            """Set saturation limits for signed fixed point with specified width.

            Parameters
            ----------
            bits: int
                Number of bits in fixed point int representation.
            """
            self.max_value = 2**(bits-1)-1
            self.min_value = -2**(bits-1)

        @property
        def name(self):
            """Saturation policy name."""
            return type(self).__name__

        def __call__(self, value):
            """Saturate given integer number according to saturation policy.

            Implement this function in subclass to create new saturation policy.

            Parameters
            ----------
            value: int
                Integer value to saturate.

            Return
            ------
            int
                Result of saturation.
            """
            raise NotImplementedError

    class Exception(SatBase):  # noqa A001
        """Raise exception if saturation occurs."""

        def __call__(self, value):
            if value > self.max_value or value < self.min_value:
                raise ValueError(f"SatPolicy.Exception: value {value} is outside [{self.min_value}, {self.max_value}].")
            return value

    class Saturation(SatBase):
        """Saturate number if lover or higher limit is hit."""

        def __call__(self, value):
            if value > self.max_value:
                return self.max_value
            if value < self.min_value:
                return self.min_value
            return value

    class Wrap(SatBase):
        """Wrap int representation of number if lover or higher limit is hit."""

        def __init__(self, bits):
            """Create wrap saturation policy. Int representation is wrapped according to signed integer rules.

            Parameters
            ----------
            bits: int
                Fixed point width.
            """
            super(SatPolicy.Wrap, self).__init__(bits)
            self.mask = 2**bits - 1

        def __call__(self, value):
            value &= self.mask
            if value > self.max_value:
                value -= self.mask + 1
            return value

#
# Fixed point class
#


class FixedPointType:
    """Fixed-point signed type with specified width, number of fractional bits, quantization and saturation policies.

    Attributes
    ----------
    bits: int
        Fixed-point type width.
    fracbits: int
        Number of fractional bits.
    sat: SatPolicy.SatBase
        Saturation policy.
    quant: QuantPolicy.QuantBase
        Quantization policy.
    """

    def __init__(self, bits, fracbits, quant, sat):
        """Create fixed point type.

        Parameters
        ----------
        bits: int
            Fixed point type width.
        fracbits: int
            Number of fractional bits.
        quant: QuantPolicy.QuantBase
            Quantization policy.
        sat: SatPolicy.SatBase
            Saturation policy.
        """
        # checks
        if not isinstance(bits, int) or not isinstance(fracbits, int) or bits < 1:
            raise ValueError("bits and fracbits arguments must be integers, bits number must be positive.")
        if not issubclass(quant, QuantPolicy.QuantBase):
            raise ValueError("quant must be subclass of QuantBase.QuantBase")
        if not issubclass(sat, SatPolicy.SatBase):
            raise ValueError("sat must be subclass of SatPolicy.SatBase")
        # type definition fields
        self.bits = bits
        self.fracbits = fracbits
        self.quant = quant(fracbits)
        self.sat = sat(bits)

    def __eq__(self, other):
        """Fixed-point types are equal if they have the same width, number of fractional bits, saturation and quantization policies."""
        return self.bits == other.bits and \
            self.fracbits == other.fracbits and \
            type(self.quant) == type(other.quant) and \
            type(self.sat) == type(other.sat)  # noqa: E721

    def reduce(self, value):
        """Reduce given value according to quantization and saturation policy to int representation of fixed-point type.

        Parameters
        ----------
        value: float or FixedPointBase
            Value to reduce.

        Return
        ------
        int
            Int representation of fixed point.
        """
        return self.sat(self.quant(value))

    def __call__(self, value=None, int_value=None):
        """Reduce given value to given fixed point type.

        Parameters
        ----------
        value: float or FixedPointBase
            Value to reduce.
        int_value: int
            Int representation of fixed point. Must be None if value is specified.

        Return
        ------
        FixedPoint
           Fixed point of given type.
        """
        return FixedPoint(value, self, int_value)

    def __repr__(self):
        """Return string representation if fixed point type."""
        return f"fixpt({self.bits},{self.fracbits},{self.quant.name},{self.sat.name})"


class FixedPointBase:
    """Fixed point value of arbitrary (automatically adjusted) precision. It is used to represent intermedia calculations results.

    Attributes
    ----------
    fracbits: int
        Number of fracbits.
    """

    def __init__(self, int_value, fracbits):
        """Create fixed point value with arbitrary precision.

        Parameters
        ----------
        int_value: int
            Int representation of fixed point value.
        fracbits: int
            Number of fractional bits.
        """
        if not isinstance(int_value, int) or not isinstance(fracbits, int):
            raise TypeError(f"int_value and fracbits arguments must be integer. Received {int_value} ({type(int_value)}) and {fracbits} ({type(fracbits)})")
        self._int = int_value
        self.fracbits = fracbits

    @property
    def float(self):  # noqa A003
        """Return float value corresponding to fixed-point."""
        return self._int / (2**self.fracbits)

    @property
    def int(self):  # noqa A003
        """Return int representation of fixed-point."""
        return self._int

    def __add__(self, other):
        other_lshift = self.fracbits - other.fracbits
        if other_lshift >= 0:
            int_sum = self._int + (other._int << other_lshift)
            return FixedPointBase(int_sum, self.fracbits)
        else:
            int_sum = other._int + (self._int << (-other_lshift))
            return FixedPointBase(int_sum, other.fracbits)

    def __radd__(self, other):
        return self.__add__(self, other)

    def __iadd__(self, other):
        other_lshift = self.fracbits - other.fracbits
        if other_lshift >= 0:
            self._int += other._int << other_lshift
        else:
            self._int = other._int + (self._int << (-other_lshift))
            self.fracbits = other.fracbits
        return self

    def __sub__(self, other):
        other_lshift = self.fracbits - other.fracbits
        if other_lshift >= 0:
            int_sum = self._int - (other._int << other_lshift)
            return FixedPointBase(int_sum, self.fracbits)
        else:
            int_sum = (self._int << (-other_lshift)) - other._int
            return FixedPointBase(int_sum, other.fracbits)

    def __rsub__(self, other):
        other_lshift = self.fracbits - other.fracbits
        if other_lshift >= 0:
            int_sum = (other._int << other_lshift) - self._int
            return FixedPointBase(int_sum, self.fracbits)
        else:
            int_sum = other._int - (self._int << (-other_lshift))
            return FixedPointBase(int_sum, other.fracbits)

    def __isub__(self, other):
        other_lshift = self.fracbits - other.fracbits
        if other_lshift >= 0:
            self._int -= other._int << other_lshift
        else:
            self._int = (self._int << (-other_lshift)) - other._int
            self.fracbits = other.fracbits
        return self

    def __mul__(self, other):
        fracbits = self.fracbits + other.fracbits
        return FixedPointBase(self._int * other._int, fracbits)

    def __rmul(self, other):
        return self.__mul__(self, other)

    def __imul__(self, other):
        self.fracbits += other.fracbits
        self._int *= other._int
        return self

    def __neg__(self):
        return FixedPointBase(-self._int, self.fracbits)

    def __lshift__(self, other):
        return FixedPointBase(self._int, self.fracbits - other)

    def __rshift__(self, other):
        return FixedPointBase(self._int, self.fracbits + other)

    def __repr__(self):
        return f"FixedPointBase({self.float}, {self.fracbits})"

    def __str__(self):
        return str(self.float)


class FixedPoint(FixedPointBase):
    """Fixed-point value with specified fiaxed-point type.

    Attributes
    ----------
    fptype: FixedPointType
        Fixed-point type instance.
    """

    def __init__(self, value=None, fptype=None, int_value=None):
        """Create fixed point value of specific fixed-point type.

        If neither value or int_value is specified default value is zero.

        Parameters
        ----------
        value: float or FixedPointBase
            Initial value. Quantized and saturated according to fixed point type policies.
        int_value: int
            Int representation of fixed point. Must be None if value is specified.
        """
        # fixed point type
        if not isinstance(fptype, FixedPointType):
            raise TypeError("fptype argument must be provided and be FixedPointType instance.")
        self.fptype = fptype
        # initial value
        if value is not None:
            super(FixedPoint, self).__init__(fptype.reduce(value), fptype.fracbits)
        elif int_value is not None:
            super(FixedPoint, self).__init__(fptype.sat(int(int_value)), fptype.fracbits)
        else:
            super(FixedPoint, self).__init__(0, fptype.fracbits)

    def assign(self, value):
        """Assign value to fixed-point variable. Fixed-point type quantization and saturation rules are applied.

        Parameters
        ----------
        value: float or FixedPointBase
            Value to assign.
        """
        self._int = self.fptype.reduce(value)

    @FixedPointBase.float.setter
    def float(self, value):  # noqa A003
        """Assign value of fixed-point variable. Equivalent of assign() method."""
        self._int = self.fptype.reduce(float(value))

    @FixedPointBase.int.setter
    def int(self, value):  # noqa A003
        """Set int representation of fixed-point variable. Saturation rules are applied."""
        self._int = self.fptype.sat(int(value))

    def __iadd__(self, other):
        value = self.__add__(other)
        self._int = self.fptype.reduce(value)
        return self

    def __isub__(self, other):
        value = self.__sub__(other)
        self._int = self.fptype.reduce(value)
        return self

    def __imul__(self, other):
        value = self.__mul__(other)
        self._int = self.fptype.reduce(value)
        return self

    def __ilshift(self, other):
        value = self.__lshift__(other)
        self._int = self.fptype.reduce(value)
        return self

    def __irshift(self, other):
        value = self.__rshift__(other)
        self._int = self.fptype.reduce(value)
        return self

    def __repr__(self):
        return f"FixedPoint({self.float}, {self.fptype})"
