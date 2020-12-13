// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"math"
)

func NewTan(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  tan,
		df: tanDeriv,
	}
}

func NewTanh(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  tanh,
		df: tanhDeriv,
	}
}

func NewSigmoid(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  sigmoid,
		df: sigmoidDeriv,
	}
}

func NewHardSigmoid(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  hardSigmoid,
		df: hardSigmoidDeriv,
	}
}

func NewHardTanh(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  hardTanh,
		df: hardTanhDeriv,
	}
}

func NewReLU(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  relu,
		df: reluDeriv,
	}
}

func NewSoftsign(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  softsign,
		df: softsignDeriv,
	}
}

func NewCos(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  func(i, j int, v float64) float64 { return math.Cos(v) },
		df: func(i, j int, v float64) float64 { return -math.Sin(v) },
	}
}

func NewSin(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  func(i, j int, v float64) float64 { return math.Sin(v) },
		df: func(i, j int, v float64) float64 { return math.Cos(v) },
	}
}

func NewExp(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  func(i, j int, v float64) float64 { return math.Exp(v) },
		df: func(i, j int, v float64) float64 { return math.Exp(v) },
	}
}

func NewLog(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  safeLog,
		df: safeLogDeriv,
	}
}

func NewNeg(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  func(i, j int, v float64) float64 { return -v },
		df: func(i, j int, v float64) float64 { return -1.0 },
	}
}

func NewReciprocal(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  func(i, j int, v float64) float64 { return 1.0 / v },
		df: func(i, j int, v float64) float64 { return -1.0 / (v * v) },
	}
}

func NewAbs(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  func(i, j int, v float64) float64 { return math.Abs(v) },
		df: absDeriv,
	}
}

func NewMish(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  mish,
		df: mishDeriv,
	}
}

func NewGeLU(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  gelu,
		df: geluDeriv,
	}
}

func NewSqrt(x Operand) *UnaryElementwise {
	return &UnaryElementwise{
		x:  x,
		f:  func(i, j int, v float64) float64 { return math.Sqrt(v) },
		df: func(i, j int, v float64) float64 { return 0.5 * math.Pow(v, -0.5) },
	}
}

func absDeriv(i, j int, v float64) float64 {
	if v < 0 {
		return -1
	} else if v > 0 {
		return 1
	} else {
		return 0 // undefined
	}
}

// safeLog is a simple work-around that make the math.Log() safe for zero or negative values
func safeLog(i, j int, v float64) float64 {
	if v > 0.0 {
		return math.Log(v)
	} else if v == 0.0 {
		return math.Log(1.0e-08)
	} else {
		panic("ag: invalid log for negative values")
	}
}

func safeLogDeriv(i, j int, v float64) float64 {
	if v > 0.0 {
		return 1.0 / v
	} else if v == 0.0 {
		return 1.0 / 1.0e-08
	} else {
		panic("ag: invalid log for negative values")
	}
}

func tan(i, j int, v float64) float64 {
	return math.Tan(v)
}

func tanDeriv(i, j int, v float64) float64 {
	return 1.0 / square(i, j, math.Cos(v))
}

func square(i, j int, v float64) float64 {
	return v * v
}

func tanh(i, j int, v float64) float64 {
	return math.Tanh(v)
}

func tanhDeriv(i, j int, v float64) float64 {
	return 1.0 - math.Pow(math.Tanh(v), 2.0)
}

func sigmoid(i, j int, v float64) float64 {
	return 1.0 / (1 + math.Exp(-v))
}

func sigmoidDeriv(i, j int, v float64) float64 {
	fx := sigmoid(i, j, v)
	return fx * (1.0 - fx)
}

func hardSigmoid(i, j int, v float64) float64 {
	if v > 2.5 {
		return 1.0
	} else if v < -2.5 {
		return 0.0
	} else {
		return 0.2*v + 0.5
	}
}

func hardSigmoidDeriv(i, j int, v float64) float64 {
	if v < 2.5 && v > -2.5 {
		return 0.2
	}
	return 0.0
}

func hardTanh(i, j int, v float64) float64 {
	if v > 1.0 {
		return 1.0
	} else if v < -1.0 {
		return -1.0
	} else {
		return v
	}
}

func hardTanhDeriv(i, j int, v float64) float64 {
	if v < 1.0 && v > -1.0 {
		return 1.0
	}
	return 0.0
}

func relu(i, j int, v float64) float64 {
	return math.Max(0.0, v)
}

func reluDeriv(i, j int, v float64) float64 {
	if v >= 0.0 {
		return 1.0
	}
	return 0.0
}

func softsign(i, j int, v float64) float64 {
	return v / (1.0 + math.Abs(v))
}

func softsignDeriv(i, j int, v float64) float64 {
	return math.Pow(1.0-math.Abs(softsign(i, j, v)), 2.0)
}

func celu(i, j int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * (math.Exp(v/alpha[0]) - 1)
	} else if v > 0 {
		return v
	}
	return 0
}

func celuDeriv(i, j int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return math.Exp(v / alpha[0])
	} else if v > 0 {
		return 1
	}
	return 0
}

func elu(i, j int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * (math.Exp(v) - 1)
	} else if v > 0 {
		return v
	}
	return 0
}

func eluDeriv(i, j int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * math.Exp(v)
	} else if v > 0 {
		return 1
	}
	return 0
}

func leakyReLU(i, j int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * v // slope * v
	} else if v > 0 {
		return v
	}
	return 0
}

func leakyReLUDeriv(i, j int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] // slope
	} else if v > 0 {
		return 1
	}
	return 0
}

// alpha[0] is the alpha
// alpha[1] is the scale
func selu(i, j int, v float64, alpha ...float64) float64 {
	scale := alpha[1]
	if v <= 0 {
		return scale * alpha[0] * (math.Exp(v) - 1)
	} else if v > 0 {
		return scale * v
	}
	return 0
}

// alpha[0] is the alpha
// alpha[1] is the scale
func seluDeriv(i, j int, v float64, alpha ...float64) float64 {
	scale := alpha[1]
	if v <= 0 {
		return scale * alpha[0] * math.Exp(v)
	} else if v > 0 {
		return scale
	}
	return 0
}

func softPlus(i, j int, v float64, alpha ...float64) float64 {
	threshold := alpha[1]
	beta := alpha[0]
	if v <= threshold {
		return (1 / beta) * math.Log(1+math.Exp(beta*v))
	} else if v > threshold {
		return v
	}
	return 0
}

func softPlusDeriv(i, j int, v float64, alpha ...float64) float64 {
	threshold := alpha[1]
	beta := alpha[0]
	if v <= threshold {
		return math.Exp(v*beta) / (math.Exp(v*beta) + 1)
	} else if v > threshold {
		return 1
	}
	return 0
}

func softShrink(i, j int, v float64, alpha ...float64) float64 {
	lambda := alpha[0]
	if v < -lambda {
		return v + lambda
	} else if v > lambda {
		return v - lambda
	}
	return 0
}

func softShrinkDeriv(i, j int, v float64, alpha ...float64) float64 {
	lambda := alpha[0]
	if v < -lambda {
		return 1
	} else if v > lambda {
		return 1
	}
	return 0
}

func threshold(i, j int, v float64, alpha ...float64) float64 {
	value := alpha[1]
	threshold := alpha[0]
	if v <= threshold {
		return value
	} else if v > threshold {
		return v
	}
	return 0
}

func thresholdDeriv(i, j int, v float64, alpha ...float64) float64 {
	threshold := alpha[0]
	if v <= threshold {
		return 0
	} else if v > threshold {
		return 1
	}
	return 0
}

func swish(i, j int, v float64, beta ...float64) float64 {
	return v * (1.0 / (1 + math.Exp(beta[0]*-v)))
}

func swishDeriv(i, j int, v float64, beta ...float64) float64 {
	prod := v * beta[0]
	exp := math.Exp(prod)
	return exp * (exp + prod + 1) / ((exp + 1) * (exp + 1))
}

func swishBetaDeriv(v float64, beta float64) float64 {
	prod := v * beta
	exp := math.Exp(-prod)
	return (v * v * exp) / ((exp + 1) * (exp + 1))
}

// Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function" by Diganta Misra, 2019.
// (https://arxiv.org/pdf/1908.08681.pdf)
func mish(i, j int, v float64) float64 {
	return v * math.Tanh(math.Log(1+math.Exp(v)))
}

func mishDeriv(i, j int, v float64) float64 {
	exp := math.Exp(v)
	exp2 := math.Exp(2 * v)
	exp3 := math.Exp(3 * v)
	omega := 4.0*(v+1.0) + 4.0*exp2 + exp3 + exp*(4.0*v+6.0)
	delta := 2*exp + exp2 + 2.0
	return exp * (omega / (delta * delta))
}

func gelu(i, j int, v float64) float64 {
	return 0.5 * v * (1.0 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*math.Pow(v, 3.0))))
}

func geluDeriv(i, j int, x float64) float64 {
	x3 := math.Pow(x, 3)
	return 0.5*math.Tanh(0.0356774*x3+0.797885*x) +
		(0.0535161*x3+0.398942*x)*
			math.Pow(1.0/math.Cosh(0.0356774*x3+0.797885*x), 2) + 0.5
}
