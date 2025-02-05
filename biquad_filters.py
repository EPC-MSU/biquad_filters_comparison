import numpy as np
import scipy.signal as sig 
import control as ctrl
from dataclasses import dataclass
from tabulate import tabulate

import fixpt
        
class BiQuad(ctrl.TransferFunction):
    def __init__(self, b, a, dt):
        super(BiQuad, self).__init__(b, a, dt = dt)
        # filter state
        self._input_hist = [ 0.0, 0.0 ]
        self._output_hist = [ 0.0, 0.0 ]

    def set_state(self, input_hist = [0.0, 0.0], output_hist = [0.0, 0.0]):
        self._input_hist = input_hist
        self._output_hist = output_hist

    def step(self, u, convert = False):
        # get paramteres
        a = self.den[0][0] 
        b = self.num[0][0]
        # calculate output
        y = b[0] * u
        for l in range(2):
            y += b[l+1] * self._input_hist[l]
            y -= a[l+1] * self._output_hist[l]
        # state update
        self._input_hist.insert(0, u)
        self._input_hist.pop(-1)
        self._output_hist.insert(0, y)
        self._output_hist.pop(-1)
        # return output
        return y

    def output(self, u, convert = False):
        # prepare output buffer
        N = len(u)
        tout = np.ndarray(N, dtype = np.float64)
        yout = np.ndarray(N, dtype = np.float64 if convert else np.int32) # TODO enshure that 32 bits is enought
        # simulation
        for k in range(N):
            tout[k] = self.dt * k
            yout[k] = self.step(u[k], convert = convert)
        return tout, yout


class BiQuadDF1(BiQuad):

    @dataclass 
    class Config:
        BASE_FREQ: int
        SIGNAL_NBITS : int
        A1_FRACBITS : int
        A2_FRACBITS : int
        B_FRACBITS : int
        STATE_FRACBITS: int
        A_NBITS : int
        B_NBITS : int
        STATE_NBITS: int
        QUANT_POLICY: object

    @dataclass 
    class Registers:
        a1 : fixpt.FixedPoint
        a2 : fixpt.FixedPoint
        b1 : fixpt.FixedPoint
        b2 : fixpt.FixedPoint
        b3 : fixpt.FixedPoint

        def __init__(self, config):
            b_fptype = fixpt.FixedPointType(config.B_NBITS, config.B_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception)
            self.b0 = fixpt.FixedPoint(0.0, b_fptype)
            self.b1 = fixpt.FixedPoint(0.0, b_fptype)
            self.b2 = fixpt.FixedPoint(0.0, b_fptype)
            self.a1 = fixpt.FixedPoint(0.0, fixpt.FixedPointType(config.A_NBITS, config.A1_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception))
            self.a2 = fixpt.FixedPoint(0.0, fixpt.FixedPointType(config.A_NBITS, config.A2_FRACBITS, fixpt.QuantPolicy.Round, fixpt.SatPolicy.Exception))

        def from_tf(self, b, a):
            self.b0.float = b[0]
            self.b1.float = b[1]
            self.b2.float = b[2]
            self.a1.float = a[1]
            self.a2.float = a[2]
            return self

        def from_raw(self, b1, b2, b3, a1, a2):
            self.b0.int = b0
            self.b1.int = b1
            self.b2.int = b2
            self.a1.int = a1
            self.a2.int = a2
            return self

        def to_raw(self):
            return self.b0.int, self.b1.int, self.b2.int, self.a1.int, self.a2.int

        def __repr__(self):
            table = []
            for attr in [ 'b0', 'b1', 'b2', 'a1', 'a2' ]:
                value = getattr(self, attr)
                table.append([ attr, value.float, value.int ])
            desc = 'BiQuadDF1.Registers\n'
            desc += tabulate(table, headers = [ 'Register', 'Value', 'Raw value' ])
            return desc
        
    def __init__(self, registers, config):
        # extract registers
        if not isinstance(registers, BiQuadDF1.Registers):
            raise TypeError('BiQuadDF1.Registers object was expected')
        self._b_fp = [ registers.b0, registers.b1, registers.b2 ]
        self._a_fp = [ registers.a1, registers.a2 ]
        # states tySTA
        self._input_fptype = fixpt.FixedPointType(config.SIGNAL_NBITS, 0, fixpt.QuantPolicy.Exception, fixpt.SatPolicy.Exception)
        self._state_fptype = fixpt.FixedPointType(config.STATE_NBITS, config.STATE_FRACBITS, config.QUANT_POLICY, fixpt.SatPolicy.Saturation)
        self._output_fptype = fixpt.FixedPointType(config.SIGNAL_NBITS, 0, config.QUANT_POLICY, fixpt.SatPolicy.Saturation)
        # construct parent object
        super(BiQuadDF1, self).__init__(b = [v.float for v in self._b_fp], a = [1.0] + [v.float for v in self._a_fp], dt = 1/config.BASE_FREQ)
        # filter state
        self._input_hist = [ self._input_fptype(0.0), self._input_fptype(0.0) ]
        self._output_hist = [ self._state_fptype(0.0), self._state_fptype(0.0) ]

    def set_state(self, input_hist = [0.0, 0.0], output_hist = [0.0, 0.0]):
        for s, v in zip(self._input_hist, input_hist):
            s.float = v 
        for s, v in zip(self._output_hist, output_hist):
            s.float = v

    def set_state_raw(self, input_hist = [0, 0], output_hist = [0, 0]):
        for s, v in zip(self._input_hist, input_hist):
            s.int = v
        for s, v in zip(self._output_hist, output_hist):
            s.int = v

    def step(self, u, convert = False):
        # input
        u_fp = self._input_fptype(float(u)) if convert else self._input_fptype(int_value = u)
        # calulate output 
        # TODO: accum range check
        accum = self._b_fp[0] * u_fp
        for l in range(2):
            accum += self._b_fp[l+1] * self._input_hist[l]
            accum -= self._a_fp[l] * self._output_hist[l]
        # quantization
        y_fp = self._output_fptype(accum)
        # state update
        self._input_hist.insert(0, u_fp)
        self._input_hist.pop(-1)
        self._output_hist.insert(0, self._state_fptype(accum))
        self._output_hist.pop(-1)
        # output
        return y_fp.float if convert else y_fp.int

class BiQuadCoupled(BiQuad):

    @dataclass
    class Config:
        BASE_FREQ: int
        SIGNAL_NBITS : int
        Q_FRACBITS : int
        ALPHA_FRACBITS : int
        STATE_FRACBITS: int
        Q_NBITS : int
        ALPHA_NBITS : int
        STATE_NBITS: int
        QUANT_POLICY: object

    @dataclass
    class Registers: 
        q0 : fixpt.FixedPoint
        q1 : fixpt.FixedPoint
        q2 : fixpt.FixedPoint
        alpha1 : fixpt.FixedPoint
        alpha2 : fixpt.FixedPoint

        def __init__(self, config):
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
            alpha = [0.0, 0.0]
            q = [0.0, 0.0, 0.0]
            # coupled form coefs
            alpha[0] = -a[1] / 2;
            if a[2] - alpha[0]**2 < 0.0:
                raise ValueError('Cannot be implemented in coupled form')
            alpha[1] = np.sqrt(a[2] - alpha[0]**2);
            q[0] = b[0];
            q[1] = q[0]*alpha[0] + b[1]/2;
            q[2] = (-b[2] + q[0]*a[2] - 2*alpha[0]*q[1]) / (2*alpha[1]);
            # covert to fixed points
            self.q0.float = q[0]
            self.q1.float = q[1]
            self.q2.float = q[2]
            self.alpha1.float = alpha[0]
            self.alpha2.float = alpha[1]
            return self

        def from_raw(self, q0, q1, q2, alpha1, alpha2):
            self.q0.int = q0
            self.q1.int = q1
            self.q2.int = q2
            self.alpha1.int = alpha1
            self.alpha2.int = alpha2
            return self

        def to_raw(self):
            return self.q0.int, self.q1.int, self.q2.int, self.alpha1.int, self.alpha2.int

        def __repr__(self):
            table = []
            for attr in [ 'q0', 'q1', 'q2', 'alpha1', 'alpha2' ]:
                value = getattr(self, attr)
                table.append([ attr, value.float, value.int ])
            desc = 'BiQuadCoupled.Registers\n'
            desc += tabulate(table, headers = [ 'Register', 'Value', 'Raw value' ])
            return desc

    def __init__(self, registers, config):
        # get coefficients
        if not isinstance(registers, BiQuadCoupled.Registers):
            raise TypeError('BiQuadCoupled.Registers object was expected')
        q_fp = [ registers.q0, registers.q1, registers.q2 ]
        alpha_fp = [ registers.alpha1, registers.alpha2 ]
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
        self._state = [ self._state_fptype(0.0), self._state_fptype(0.0) ]
        # coefficirnts
        self._alpha = alpha_fp
        self._q = q_fp

    def set_state(self, s1 = 0.0, s2 = 0.0):
        self._state[0].float = s1
        self._state[1].float = s2

    def set_state_raw(self, s1 = 0, s2 = 0):
        self._state[0].int = s1
        self._state[1].int = s2

    def step(self, u, convert = False):
         # input
        u_fp = self._input_fptype(float(u)) if convert else self._input_fptype(int_value = u)
        # calulate output 
        y_fp = fixpt.FixedPoint((self._state[0] << 1) + self._q[0] * u_fp, self._output_fptype)
        # update state
        accum0 = self._q[1] * u_fp + self._alpha[0] * self._state[0] - self._alpha[1] * self._state[1]
        accum1 = self._q[2] * u_fp + self._alpha[0] * self._state[1] + self._alpha[1] * self._state[0]
        self._state[0].assign(accum0)
        self._state[1].assign(accum1)
        # output
        return y_fp.float if convert else y_fp.int






