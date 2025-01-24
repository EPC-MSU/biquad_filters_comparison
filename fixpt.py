import math

#
# quantization methods
#

class QuantPolicy:
    class QuantBase:
        def __init__(self, fracbits):
            self.fracbits = fracbits

        @property
        def name(self):
            return type(self).__name__

    class Truncate(QuantBase):
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
                raise TypeError(f'FixedPointBase or float is expected, {type(value)} is received.')

    class TruncateToZero(QuantBase):
        def __call__(self, value):
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
                raise TypeError(f'FixedPointBase or float is expected, {type(value)} is received.')

    class Round(QuantBase):
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
                raise TypeError(f'FixedPointBase or float is expected, {type(value)} is received.')

    class Exception(QuantBase):
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
                        raise ValueError('QuantPolicy.Exception: unable to assign without rounding.')
            elif isinstance(value, float):
                int_value = value * 2**self.fracbits
                if int_value.is_integer():
                    return int(int_value)
                else:
                    raise ValueError(f'QuantPolicy.Exception: {value} can not be exactly represented as fixed point with {self.fracbits} fracbits')
            else:
                raise TypeError(f'FixedPointBase or float is expected, {type(value)} is received.')

# 
# saturation methods
#

class SatPolicy:
    class SatBase:
        def __init__(self, bits):
            self.max_value = 2**(bits-1)-1
            self.min_value = -2**(bits-1)

        @property
        def name(self):
            return type(self).__name__

    class Exception(SatBase):
        def __call__(self, value):
            if value > self.max_value or value < self.min_value:
                raise ValueError(f'SatPolicy.Exception: value {value} is outside [{self.min_value}, {self.max_value}].')
            return value

    class Saturation(SatBase):
        def __call__(self, value):
            if value > self.max_value:
                return self.max_value
            if value < self.min_value:
                return self.min_value
            return value

    class Wrap(SatBase):
        def __init__(self, bits):
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
    def __init__(self, bits, fracbits, quant, sat):
        # checks
        if not isinstance(bits, int) or not isinstance(fracbits, int) or bits < 1:
            raise ValueError('bits and fracbits arguments must be integers, bits number must be positive.')
        if not issubclass(quant, QuantPolicy.QuantBase):
            raise ValueError('quant must be subclass of QuantBase.QuantBase')
        if not issubclass(sat, SatPolicy.SatBase):
            raise ValueError('sat must be subclass of SatPolicy.SatBase')
        # type definition fields
        self.bits = bits
        self.fracbits = fracbits
        self.quant = quant(fracbits)
        self.sat = sat(bits)

    def __eq__(self, other):
        return self.bits == other.bits and \
               self.fracbits == other.fracbits and \
               type(self.quant) == type(other.quant) and \
               type(self.sat) == type(other.sat)

    def reduce(self, value):
        return self.sat(self.quant(value))

    def __call__(self, value = None, int_value = None):
        return FixedPoint(value, self, int_value)

    def __repr__(self):
        return f'fixpt({self.bits},{self.fracbits},{self.quant.name},{self.sat.name})'

class FixedPointBase:
    def __init__(self, int_value, fracbits):
        if not isinstance(int_value, int) or not isinstance(fracbits, int):
            raise TypeError(f'int_value and fracbits arguments must be integer. Received {int_value} ({type(int_value)}) and {fracbits} ({type(fracbits)})')
        self._int = int_value
        self.fracbits = fracbits

    @property
    def float(self):
        return self._int / (2**self.fracbits)

    @property
    def int(self):
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
        return __add__(self, other)

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
        return __mul__(self, other)

    def __imul__(self, other):
        self.fracbits +=  other.fracbits
        self._int *= other._int
        return self

    def __neg__(self):
        return FixedPointBase(-self._int, self.fracbits)

    def __lshift__(self, other):
        return FixedPointBase(self._int, self.fracbits - other)

    def __rshift__(self, other):
        return FixedPointBase(self._int, self.fracbits + other)

    def __repr__(self):
        return f'FixedPointBase({self.float}, {self.fracbits})'

    def __str__(self):
        return str(self.float)


class FixedPoint(FixedPointBase):

    def __init__(self, value = None, fptype = None, int_value = None):
        # fixed point type
        if not isinstance(fptype, FixedPointType):
            raise TypeError('fptype argument must be provided and be FixedPointType instance.')
        self.fptype = fptype
        # initial value
        if value is not None:
            super(FixedPoint, self).__init__(fptype.reduce(value), fptype.fracbits)
        elif int_value is not None: 
            super(FixedPoint, self).__init__(fptype.sat(int(int_value)), fptype.fracbits)
        else:
            super(FixedPoint, self).__init__(0, fptype.fracbits)

    def assign(self, value):
        self._int = self.fptype.reduce(value)

    @FixedPointBase.float.setter
    def float(self, value):
        self._int = self.fptype.reduce(float(value))

    @FixedPointBase.int.setter
    def int(self, value):
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
        return f'FixedPoint({self.float}, {self.fptype})'





