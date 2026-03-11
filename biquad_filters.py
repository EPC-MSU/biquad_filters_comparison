"""Simulation of different realization of biquadrat filters.

Biquad filter realizations classes:
* BiQuad. Float-point Direct Form I
* BiQuadDF1. Fixed-point Direct Form I.
* BiQuadCoupled. Fixed-point coupled form.

Realization variants extends control.TransferFunction class.
* Underlying transfer function coefficients are set taking in account quantization.
* Standard simulation works with these coefficients.
* To fully take in account realization-specific effects use output() method.
"""

from dataclasses import dataclass, field

import control as ctrl

import fixpt

import numpy as np

from tabulate import tabulate


class BiQuad(ctrl.TransferFunction):
    """Float-point Direct Form I biquadratic filter realization."""

    def __init__(self, b, a, dt):
        """Create lloat-point biquadratic filter realization.

        Paramters
        ---------
        a: array_like
            Denominator coefficients.
        b: array_like
            Numerator coefficients.
        dt: float
            Sample period.
        """
        super(BiQuad, self).__init__(b, a, dt=dt)
        # filter state
        self._input_hist = [0.0, 0.0]
        self._output_hist = [0.0, 0.0]

    def set_state(self, input_hist=[0.0, 0.0], output_hist=[0.0, 0.0]):
        """Set DF1 filter state.

        Parameters
        ----------
        input_hist: array_like
            Filter input history (two elements)
        input_output: array_like
            Filter output history (two elements).
        """
        self._input_hist = input_hist
        self._output_hist = output_hist

    def step(self, u, convert=False):
        """Perform one simulation step.

        Parameters
        ----------
        u: float
            Input value.
        convert: bool, default False
            For fixed-point realization if set to False return output as fixed-point int representation.
            If set to True return float. Has no effect on float-point filters.

        Return
        ------
        y: float or int
            Filter output as float (convert is set to True) or int representation of fixed-point
            (convert is set to False).
        """
        # get paramteres
        a = self.den[0][0]
        b = self.num[0][0]
        # calculate output
        y = b[0] * u
        for k in range(2):
            y += b[k+1] * self._input_hist[k]
            y -= a[k+1] * self._output_hist[k]
        # state update
        self._input_hist.insert(0, u)
        self._input_hist.pop(-1)
        self._output_hist.insert(0, y)
        self._output_hist.pop(-1)
        # return output
        return y

    def output(self, u, convert=False):
        """Simulate filter.

        Parameters
        ----------
        u: array_like
            Filter input.
        convert: bool, default False
            If set to True return float array as system output. Otherwise if system has fixed-point realization
            return int representation of fixed-point output.

        Return
        ------
        numpy.ndarray
            Filter output. Dtype is np.float if convert is True and np.int32 otherwise.
        """
        # prepare output buffer
        N = len(u)
        yout = np.ndarray(N, dtype=np.float64 if convert else np.int32)  # TODO enshure that 32 bits is enought
        # simulation
        for k in range(N):
            yout[k] = self.step(u[k], convert=convert)
        return yout


class BiQuadDF1(BiQuad):
    """Fixed-point Direct Form I biquadratic filter realization.

                      -1       -2
             b0 + b1 z   + b2 z
    F(z) = -----------------------
                      -1       -2
             1  + a1 z   + a2 z
    """

    @dataclass
    class Config:
        """Configuration parameters of fixed-point Direct Form I biquadratic filter realization."""

        BASE_FREQ: int = field(metadata={"doc": "Filter sample rate (Hz)"})
        SIGNAL_NBITS: int = field(metadata={"doc": "Input and output signal width."})
        A1_FRACBITS: int = field(metadata={"doc": "A1 coefficient number of fractional bits."})
        A2_FRACBITS: int = field(metadata={"doc": "A2 coefficient number of fractional bits."})
        B_FRACBITS: int = field(metadata={"doc": "B0, B1, B2 coefficients number of fractional bits."})
        STATE_FRACBITS: int = field(metadata={"doc": "Additional fractional bits for state variables."})
        A_NBITS: int = field(metadata={"doc": "A1, A2 coefficients width."})
        B_NBITS: int = field(metadata={"doc": "B0, B1, B2 coefficients width."})
        STATE_NBITS: int = field(metadata={"doc": "State variables width."})
        QUANT_POLICY: fixpt.QuantPolicy.QuantBase = field(metadata={"doc": "State and output quantization policy."})

    @dataclass
    class Registers:
        """Fixed-point Direct Form I biquadratic filter coefficients as fixed-point values."""

        a1: fixpt.FixedPoint
        a2: fixpt.FixedPoint
        b0: fixpt.FixedPoint
        b1: fixpt.FixedPoint
        b2: fixpt.FixedPoint

        def __init__(self, config):
            """Create filter coefficients structure according to filter configuration parameters.

            Round quantization policy is used.

            Parameters
            ----------
            config: BiQuadDF1.Config
                Filter configuration parameters which specifies width and fractional bit number of fixed-point values.
            """
            b_fptype = fixpt.FixedPointType(config.B_NBITS, config.B_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception)
            self.b0 = fixpt.FixedPoint(0.0, b_fptype)
            self.b1 = fixpt.FixedPoint(0.0, b_fptype)
            self.b2 = fixpt.FixedPoint(0.0, b_fptype)
            self.a1 = fixpt.FixedPoint(0.0, fixpt.FixedPointType(config.A_NBITS, config.A1_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception))
            self.a2 = fixpt.FixedPoint(0.0, fixpt.FixedPointType(config.A_NBITS, config.A2_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception))

        def from_tf(self, b, a):
            """Assign filter realization coefficients from transfer function.

            Parameters
            ----------
            b: array_like
                Transfer function numerator.
            a: array_like
                Transfer function denominator. Leading element must be 1.0
            """
            self.b0.float = b[0]
            self.b1.float = b[1]
            self.b2.float = b[2]
            self.a1.float = a[1]
            self.a2.float = a[2]
            return self

        def from_raw(self, b0, b1, b2, a1, a2):
            """Assign filter realization coefficients from their int representation."""
            self.b0.int = b0
            self.b1.int = b1
            self.b2.int = b2
            self.a1.int = a1
            self.a2.int = a2
            return self

        def to_raw(self):
            """Get int representation of filter realization coefficients.

            Return
            ------
            tuple
                Tuple of filter coefficients: (b0, b1, b2, a1, a2).
            """
            return self.b0.int, self.b1.int, self.b2.int, self.a1.int, self.a2.int

        def __repr__(self):
            table = []
            for attr in ["b0", "b1", "b2", "a1", "a2"]:
                value = getattr(self, attr)
                table.append([attr, value.float, value.int])
            desc = "BiQuadDF1.Registers\n"
            desc += tabulate(table, headers=["Register", "Value", "Raw value"])
            return desc

    def __init__(self, registers, config):
        """Create fixed-point Direct Form I biquadratic filter.

        Parameters
        ----------
        registers: BiQuadDF1.Registers
            Filter coefficients.
        config: BiQuadDF1.Config
            Filter configuration parameters.
        """
        # extract registers
        if not isinstance(registers, BiQuadDF1.Registers):
            raise TypeError("BiQuadDF1.Registers object was expected")
        self._b_fp = [registers.b0, registers.b1, registers.b2]
        self._a_fp = [registers.a1, registers.a2]
        # states tySTA
        self._input_fptype = fixpt.FixedPointType(config.SIGNAL_NBITS, 0, fixpt.QuantPolicy.Exception, fixpt.SatPolicy.Exception)
        self._state_fptype = fixpt.FixedPointType(config.STATE_NBITS, config.STATE_FRACBITS, config.QUANT_POLICY, fixpt.SatPolicy.Saturation)
        self._output_fptype = fixpt.FixedPointType(config.SIGNAL_NBITS, 0, config.QUANT_POLICY, fixpt.SatPolicy.Saturation)
        # construct parent object
        super(BiQuadDF1, self).__init__(b=[v.float for v in self._b_fp], a=[1.0] + [v.float for v in self._a_fp], dt=1/config.BASE_FREQ)
        # filter state
        self._input_hist = [self._input_fptype(0.0), self._input_fptype(0.0)]
        self._output_hist = [self._state_fptype(0.0), self._state_fptype(0.0)]

    def set_state(self, input_hist=[0.0, 0.0], output_hist=[0.0, 0.0]):
        """Set DF1 filter state. Float values are converted to fixed-points.

        Parameters
        ----------
        input_hist: array_like
            Filter input history (two elements)
        input_output: array_like
            Filter output history (two elements).
        """
        for s, v in zip(self._input_hist, input_hist):
            s.float = v
        for s, v in zip(self._output_hist, output_hist):
            s.float = v

    def set_state_raw(self, input_hist=[0, 0], output_hist=[0, 0]):
        """Set DF1 filter fixed-point state from its int representation.

        Parameters
        ----------
        input_hist: array_like
            Filter input history (two elements)
        input_output: array_like
            Filter output history (two elements).
        """
        for s, v in zip(self._input_hist, input_hist):
            s.int = v
        for s, v in zip(self._output_hist, output_hist):
            s.int = v

    def step(self, u, convert=False):
        # input
        u_fp = self._input_fptype(float(u)) if convert else self._input_fptype(int_value=u)
        # calulate output
        # TODO: accum range check
        accum = self._b_fp[0] * u_fp
        for k in range(2):
            accum += self._b_fp[k+1] * self._input_hist[k]
            accum -= self._a_fp[k] * self._output_hist[k]
        # quantization
        state = self._state_fptype(accum)
        y_fp = self._output_fptype(state)
        # state update
        self._input_hist.insert(0, u_fp)
        self._input_hist.pop(-1)
        self._output_hist.insert(0, state)
        self._output_hist.pop(-1)
        # output
        return y_fp.float if convert else y_fp.int


class BiQuadCoupled(BiQuad):
    """Fixed-point coupled realization of biquadratical filter.

                                  -1          2     2                    -2
             q0 + (q1 - 2 q0 a1) z   + (q0 (a1  + a2 ) - q1 a1 - q2 a2) z
    F(z) =  ---------------------------------------------------------------
                                  -1      2     2   -2
                        1 - 2 a1 z   + (a1  + a2 ) z
    """

    @dataclass
    class Config:
        """Fixed-point coupled realization of biquadratical filter configuration parameters."""

        BASE_FREQ: int = field(metadata={"doc": "Sample frequency (Hz)"})
        SIGNAL_NBITS: int = field(metadata={"doc": "Input and output fixed-point width."})
        Q_FRACBITS: int = field(metadata={"doc": "Number of fractional bits for Q0, Q1, Q2 coefficients."})
        ALPHA_FRACBITS: int = field(metadata={"doc": "Number of fractional bits for A1, A2 coefficients."})
        STATE_FRACBITS: int = field(metadata={"doc": "Number of fractional bits for filter state."})
        Q_NBITS: int = field(metadata={"doc": "Filter state width."})
        ALPHA_NBITS: int = field(metadata={"doc": "A1, A2 coefficients width."})
        STATE_NBITS: int = field(metadata={"doc": "Q0, Q1, Q2 coefficients width."})
        QUANT_POLICY: fixpt.QuantPolicy.QuantBase = field(metadata={"doc": "Quantization policy for state and output."})

    @dataclass
    class Registers:
        """Fixed-point coupled form biquadratic filter coefficients as fixed-points."""

        q0: fixpt.FixedPoint
        q1: fixpt.FixedPoint
        q2: fixpt.FixedPoint
        alpha1: fixpt.FixedPoint
        alpha2: fixpt.FixedPoint

        def __init__(self, config):
            """Create filter coefficients structure according to filter configuration parameters.

            Round quantization policy is used.

            Parameters
            ----------
            config: BiQuadCoupled.Config
                Filter configuration parameters which specifies width and fractional bit number of fixed-point values.
            """
            # types
            q_fptype = fixpt.FixedPointType(config.Q_NBITS, config.Q_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception)
            alpha_fptype = fixpt.FixedPointType(config.ALPHA_NBITS, config.ALPHA_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception)
            # zero values
            self.q0 = fixpt.FixedPoint(0.0, q_fptype)
            self.q1 = fixpt.FixedPoint(0.0, q_fptype)
            self.q2 = fixpt.FixedPoint(0.0, q_fptype)
            self.alpha1 = fixpt.FixedPoint(0.0, alpha_fptype)
            self.alpha2 = fixpt.FixedPoint(0.0, alpha_fptype)

        def from_tf(self, b, a):
            """Assign filter realization coefficients from transfer function.

            Parameters
            ----------
            b: array_like
                Transfer function numerator.
            a: array_like
                Transfer function denominator. Leading element must be 1.0
            """
            alpha = [0.0, 0.0]
            q = [0.0, 0.0, 0.0]
            # coupled form coefs
            alpha[0] = -a[1] / 2
            if a[2] - alpha[0]**2 < 0.0:
                raise ValueError("Cannot be implemented in coupled form.")
            alpha[1] = np.sqrt(a[2] - alpha[0]**2)
            q[0] = b[0]
            q[1] = q[0]*alpha[0] + b[1]/2
            q[2] = (-b[2] + q[0]*a[2] - 2*alpha[0]*q[1]) / (2*alpha[1])
            # covert to fixed points
            self.q0.float = q[0]
            self.q1.float = q[1]
            self.q2.float = q[2]
            self.alpha1.float = alpha[0]
            self.alpha2.float = alpha[1]
            return self

        def from_raw(self, q0, q1, q2, alpha1, alpha2):
            """Assign filter realization coefficients from their int representation."""
            self.q0.int = q0
            self.q1.int = q1
            self.q2.int = q2
            self.alpha1.int = alpha1
            self.alpha2.int = alpha2
            return self

        def to_raw(self):
            """Get int representation of filter realization coefficients.

            Return
            ------
            tuple
                Tuple of filter coefficients: (q0, q1, q2, alpha1, alpha2).
            """
            return self.q0.int, self.q1.int, self.q2.int, self.alpha1.int, self.alpha2.int

        def __repr__(self):
            table = []
            for attr in ["q0", "q1", "q2", "alpha1", "alpha2"]:
                value = getattr(self, attr)
                table.append([attr, value.float, value.int])
            desc = "BiQuadCoupled.Registers\n"
            desc += tabulate(table, headers=["Register", "Value", "Raw value"])
            return desc

    def __init__(self, registers, config):
        """Create fixed-point Coupled Form biquadratic filter.

        Parameters
        ----------
        registers: BiQuadCoupled.Registers
            Filter coefficients.
        config: BiQuadCoupled.Config
            Filter configuration parameters.
        """
        # get coefficients
        if not isinstance(registers, BiQuadCoupled.Registers):
            raise TypeError("BiQuadCoupled.Registers object was expected")
        q_fp = [registers.q0, registers.q1, registers.q2]
        alpha_fp = [registers.alpha1, registers.alpha2]
        # recalculate tf
        a = np.array([1.0, -2.0*alpha_fp[0].float, alpha_fp[0].float**2 + alpha_fp[1].float**2])
        b = q_fp[0].float * a \
            + 2.0*q_fp[1].float * np.array([0.0, 1.0, -alpha_fp[0].float]) \
            + 2.0*q_fp[2].float * np.array([0.0, 0.0, -alpha_fp[1].float])
        # construct parent object
        super(BiQuadCoupled, self).__init__(b, a, 1.0 / config.BASE_FREQ)
        #  fixed points dtypes
        self._input_fptype = fixpt.FixedPointType(config.SIGNAL_NBITS, 0, fixpt.QuantPolicy.Exception, fixpt.SatPolicy.Exception)
        self._state_fptype = fixpt.FixedPointType(config.STATE_NBITS, config.STATE_FRACBITS, config.QUANT_POLICY, fixpt.SatPolicy.Saturation)
        self._output_fptype = fixpt.FixedPointType(config.SIGNAL_NBITS, 0, config.QUANT_POLICY, fixpt.SatPolicy.Saturation)
        # filter state
        self._state = [self._state_fptype(0.0), self._state_fptype(0.0)]
        # coefficients
        self._alpha = alpha_fp
        self._q = q_fp

    def set_state(self, s1=0.0, s2=0.0):
        """Set coupled form biquad filter state. Float values are converted to fixed-points.

        Parameters
        ----------
        s1: float, default 0.0
            Filter state variable.
        s2: flaot, default 0.0
            Filter state variable.
        """
        self._state[0].float = s1
        self._state[1].float = s2

    def set_state_raw(self, s1=0, s2=0):
        """Set coupled form fixed-point biquad filter state from its int representation.

        Parameters
        ----------
        s1: int, default 0
            Filter state variable
        s2: int, default 0
            Filter state variable.
        """
        self._state[0].int = s1
        self._state[1].int = s2

    def step(self, u, convert=False):
        # input
        u_fp = self._input_fptype(float(u)) if convert else self._input_fptype(int_value=u)
        # calculate output
        y_fp = fixpt.FixedPoint((self._state[0] << 1) + self._q[0] * u_fp, self._output_fptype)
        # update state
        accum0 = self._q[1] * u_fp + self._alpha[0] * self._state[0] - self._alpha[1] * self._state[1]
        accum1 = self._q[2] * u_fp + self._alpha[0] * self._state[1] + self._alpha[1] * self._state[0]
        self._state[0].assign(accum0)
        self._state[1].assign(accum1)
        # output
        return y_fp.float if convert else y_fp.int
