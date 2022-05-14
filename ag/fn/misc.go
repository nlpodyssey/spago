// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"math"
)

// Tan is an operator to perform element-wise tangent.
type Tan[O Operand] struct {
	*UnaryElementwise[O]
}

// NewTan returns a new UnaryElementwise tangent function.
func NewTan[O Operand](x O) *Tan[O] {
	return &Tan[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        tan,
			df:       tanDeriv,
			operands: []O{x},
		},
	}
}

// Tanh is an operator to perform element-wise hyperbolic tangent.
type Tanh[O Operand] struct {
	*UnaryElementwise[O]
}

// NewTanh returns a new UnaryElementwise hyperbolic tangent function.
func NewTanh[O Operand](x O) *Tanh[O] {
	return &Tanh[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        tanh,
			df:       tanhDeriv,
			operands: []O{x},
		},
	}
}

// Sigmoid is an operator to perform element-wise sigmoid.
type Sigmoid[O Operand] struct {
	*UnaryElementwise[O]
}

// NewSigmoid returns a new UnaryElementwise sigmoid function.
func NewSigmoid[O Operand](x O) *Sigmoid[O] {
	return &Sigmoid[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        sigmoid,
			df:       sigmoidDeriv,
			operands: []O{x},
		},
	}
}

// HardSigmoid is an operator to perform element-wise hard sigmoid.
type HardSigmoid[O Operand] struct {
	*UnaryElementwise[O]
}

// NewHardSigmoid returns a new UnaryElementwise hard sigmoid function.
func NewHardSigmoid[O Operand](x O) *HardSigmoid[O] {
	return &HardSigmoid[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        hardSigmoid,
			df:       hardSigmoidDeriv,
			operands: []O{x},
		},
	}
}

// HardTanh is an operator to perform element-wise hard hyperbolic tangent.
type HardTanh[O Operand] struct {
	*UnaryElementwise[O]
}

// NewHardTanh returns a new UnaryElementwise hard hyperbolic tangent function.
func NewHardTanh[O Operand](x O) *HardTanh[O] {
	return &HardTanh[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        hardTanh,
			df:       hardTanhDeriv,
			operands: []O{x},
		},
	}
}

// ReLU is an operator to perform element-wise Rectified Linear Unit (ReLU)
type ReLU[O Operand] struct {
	*UnaryElementwise[O]
}

// NewReLU returns a new UnaryElementwise Rectified Linear Unit (ReLU) function.
func NewReLU[O Operand](x O) *ReLU[O] {
	return &ReLU[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        relu,
			df:       reluDeriv,
			operands: []O{x},
		},
	}
}

// Softsign is an operator to perform element-wise softsign.
type Softsign[O Operand] struct {
	*UnaryElementwise[O]
}

// NewSoftsign returns a new UnaryElementwise softsign function.
func NewSoftsign[O Operand](x O) *Softsign[O] {
	return &Softsign[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        softsign,
			df:       softsignDeriv,
			operands: []O{x},
		},
	}
}

// Cos is an operator to perform element-wise cos.
type Cos[O Operand] struct {
	*UnaryElementwise[O]
}

// NewCos returns a new UnaryElementwise cos function.
func NewCos[O Operand](x O) *Cos[O] {
	return &Cos[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        func(_, _ int, v float64) float64 { return math.Cos(v) },
			df:       func(_, _ int, v float64) float64 { return -math.Sin(v) },
			operands: []O{x},
		},
	}
}

// Sin is an operator to perform element-wise sin.
type Sin[O Operand] struct {
	*UnaryElementwise[O]
}

// NewSin returns a new UnaryElementwise sine function.
func NewSin[O Operand](x O) *Sin[O] {
	return &Sin[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Sin(v) },
			df:       func(i, j int, v float64) float64 { return math.Cos(v) },
			operands: []O{x},
		},
	}
}

// Exp is an operator to perform element-wise base-e exponential.
type Exp[O Operand] struct {
	*UnaryElementwise[O]
}

// NewExp returns a new UnaryElementwise base-e exponential function.
func NewExp[O Operand](x O) *Exp[O] {
	return &Exp[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Exp(v) },
			df:       func(i, j int, v float64) float64 { return math.Exp(v) },
			operands: []O{x},
		},
	}
}

// Log is an operator to perform element-wise natural logarithm.
type Log[O Operand] struct {
	*UnaryElementwise[O]
}

// NewLog returns a new UnaryElementwise natural logarithm function.
func NewLog[O Operand](x O) *Log[O] {
	return &Log[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        safeLog,
			df:       safeLogDeriv,
			operands: []O{x},
		},
	}
}

// Neg is an operator to perform element-wise f(x) = -x
type Neg[O Operand] struct {
	*UnaryElementwise[O]
}

// NewNeg returns a new UnaryElementwise f(x) = -x function.
func NewNeg[O Operand](x O) *Neg[O] {
	return &Neg[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return -v },
			df:       func(i, j int, v float64) float64 { return -1.0 },
			operands: []O{x},
		},
	}
}

// Reciprocal is an operator to perform element-wise reciprocal.
type Reciprocal[O Operand] struct {
	*UnaryElementwise[O]
}

// NewReciprocal returns a new UnaryElementwise reciprocal function.
func NewReciprocal[O Operand](x O) *Reciprocal[O] {
	return &Reciprocal[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return 1.0 / v },
			df:       func(i, j int, v float64) float64 { return -1.0 / (v * v) },
			operands: []O{x},
		},
	}
}

// Abs is an operator to perform element-wise absolute value function.
type Abs[O Operand] struct {
	*UnaryElementwise[O]
}

// NewAbs returns a new UnaryElementwise absolute value function.
func NewAbs[O Operand](x O) *Abs[O] {
	return &Abs[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Abs(v) },
			df:       absDeriv,
			operands: []O{x},
		},
	}
}

// Mish is an operator to perform element-wise mish.
type Mish[O Operand] struct {
	*UnaryElementwise[O]
}

// NewMish returns a new UnaryElementwise Mish function.
//
// Mish is a self-regularized non-monotonic activation function which can be
// mathematically defined as f(x) = x * tanh(softplus(x)).
//
// Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
// by Diganta Misra, 2019 (https://arxiv.org/pdf/1908.08681.pdf)
func NewMish[O Operand](x O) *Mish[O] {
	return &Mish[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        mish,
			df:       mishDeriv,
			operands: []O{x},
		},
	}
}

// GELU is an operator to perform element-wise GELU.
type GELU[O Operand] struct {
	*UnaryElementwise[O]
}

// NewGELU returns a new UnaryElementwise Gaussian Error Linear Unit (GELU) function.
func NewGELU[O Operand](x O) *GELU[O] {
	return &GELU[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        gelu,
			df:       geluDeriv,
			operands: []O{x},
		},
	}
}

// Sqrt is an operator to perform element-wise square root.
type Sqrt[O Operand] struct {
	*UnaryElementwise[O]
}

// NewSqrt returns a new UnaryElementwise square root function.
func NewSqrt[O Operand](x O) *Sqrt[O] {
	return &Sqrt[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Sqrt(v) },
			df:       func(i, j int, v float64) float64 { return 0.5 * math.Pow(v, -0.5) },
			operands: []O{x},
		},
	}
}

// Swish is an operator to perform element-wise x * sigmoid(x).
type Swish[O Operand] struct {
	*UnaryElementwise[O]
}

// NewSwish returns a new function of the form f(x) = x * sigmoid(x).
func NewSwish[O Operand](x O) *Swish[O] {
	return &Swish[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:        x,
			f:        swish,
			df:       swishDeriv,
			operands: []O{x},
		},
	}
}

// NewSiLU (Sigmoid Linear Unit) returns a new function of the form f(x) = x * sigmoid(x).
// The function in an alias of NewSwish.
func NewSiLU[O Operand](x O) *Swish[O] {
	return NewSwish[O](x)
}

func absDeriv(_, _ int, v float64) float64 {
	if v < 0 {
		return -1
	}
	if v > 0 {
		return 1
	}
	return 0 // undefined
}

// safeLog is a simple work-around that make the math.Log() safe for zero or negative values
func safeLog(_, _ int, v float64) float64 {
	if v > 0.0 {
		return math.Log(v)
	}
	if v == 0.0 {
		return math.Inf(-1)
	}
	panic("ag: invalid log for negative values")
}

func safeLogDeriv(_, _ int, v float64) float64 {
	if v > 0.0 {
		return 1.0 / v
	}
	if v == 0.0 {
		return 1.0 / 1.0e-08
	}
	panic("ag: invalid log for negative values")
}

func tan(_, _ int, v float64) float64 {
	return math.Tan(v)
}

func tanDeriv(_, _ int, v float64) float64 {
	c := math.Cos(v)
	return 1.0 / (c * c)
}

func tanh(_, _ int, v float64) float64 {
	return math.Tanh(v)
}

func tanhDeriv(_, _ int, v float64) float64 {
	return 1.0 - math.Pow(math.Tanh(v), 2.0)
}

func sigmoid(_, _ int, v float64) float64 {
	return 1.0 / (1 + math.Exp(-v))
}

func sigmoidDeriv(i, j int, v float64) float64 {
	fx := sigmoid(i, j, v)
	return fx * (1.0 - fx)
}

func hardSigmoid(_, _ int, v float64) float64 {
	if v > 2.5 {
		return 1.0
	}
	if v < -2.5 {
		return 0.0
	}
	return 0.2*v + 0.5
}

func hardSigmoidDeriv(_, _ int, v float64) float64 {
	if v < 2.5 && v > -2.5 {
		return 0.2
	}
	return 0.0
}

func hardTanh(_, _ int, v float64) float64 {
	if v > 1.0 {
		return 1.0
	}
	if v < -1.0 {
		return -1.0
	}
	return v
}

func hardTanhDeriv(_, _ int, v float64) float64 {
	if v < 1.0 && v > -1.0 {
		return 1.0
	}
	return 0.0
}

func relu(_, _ int, v float64) float64 {
	return math.Max(0.0, v)
}

func reluDeriv(_, _ int, v float64) float64 {
	if v >= 0.0 {
		return 1.0
	}
	return 0.0
}

func softsign(_, _ int, v float64) float64 {
	return v / (1.0 + math.Abs(v))
}

func softsignDeriv(i, j int, v float64) float64 {
	return math.Pow(1.0-math.Abs(softsign(i, j, v)), 2.0)
}

func celu(_, _ int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * (math.Exp(v/alpha[0]) - 1)
	}
	if v > 0 {
		return v
	}
	return 0
}

func celuDeriv(_, _ int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return math.Exp(v / alpha[0])
	}
	if v > 0 {
		return 1
	}
	return 0
}

func elu(_, _ int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * (math.Exp(v) - 1)
	}
	if v > 0 {
		return v
	}
	return 0
}

func eluDeriv(_, _ int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * math.Exp(v)
	}
	if v > 0 {
		return 1
	}
	return 0
}

func leakyReLU(_, _ int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] * v // slope * v
	}
	if v > 0 {
		return v
	}
	return 0
}

func leakyReLUDeriv(_, _ int, v float64, alpha ...float64) float64 {
	if v <= 0 {
		return alpha[0] // slope
	}
	if v > 0 {
		return 1
	}
	return 0
}

// alpha[0] is the alpha
// alpha[1] is the scale
func selu(_, _ int, v float64, alpha ...float64) float64 {
	scale := alpha[1]
	if v <= 0 {
		return scale * alpha[0] * (math.Exp(v) - 1)
	}
	if v > 0 {
		return scale * v
	}
	return 0
}

// alpha[0] is the alpha
// alpha[1] is the scale
func seluDeriv(_, _ int, v float64, alpha ...float64) float64 {
	scale := alpha[1]
	if v <= 0 {
		return scale * alpha[0] * math.Exp(v)
	}
	if v > 0 {
		return scale
	}
	return 0
}

func softPlus(_, _ int, v float64, alpha ...float64) float64 {
	threshold := alpha[1]
	beta := alpha[0]
	if v <= threshold {
		return (1 / beta) * math.Log(1+math.Exp(beta*v))
	}
	if v > threshold {
		return v
	}
	return 0
}

func softPlusDeriv(_, _ int, v float64, alpha ...float64) float64 {
	threshold := alpha[1]
	beta := alpha[0]
	if v <= threshold {
		exp := math.Exp(v * beta)
		return exp / (exp + 1)
	}
	if v > threshold {
		return 1
	}
	return 0
}

func softShrink(_, _ int, v float64, alpha ...float64) float64 {
	lambda := alpha[0]
	if v < -lambda {
		return v + lambda
	}
	if v > lambda {
		return v - lambda
	}
	return 0
}

func softShrinkDeriv(_, _ int, v float64, alpha ...float64) float64 {
	lambda := alpha[0]
	if v < -lambda {
		return 1
	}
	if v > lambda {
		return 1
	}
	return 0
}

func threshold(_, _ int, v float64, alpha ...float64) float64 {
	value := alpha[1]
	t := alpha[0]
	if v <= t {
		return value
	}
	if v > t {
		return v
	}
	return 0
}

func thresholdDeriv(_, _ int, v float64, alpha ...float64) float64 {
	t := alpha[0]
	if v <= t {
		return 0
	}
	if v > t {
		return 1
	}
	return 0
}

func swish(_, _ int, v float64) float64 {
	return v / (1 + math.Exp(-v))
}

func swishDeriv(i, j int, v float64) float64 {
	return swishBDeriv(i, j, v, 1.0)
}

func swishB(_, _ int, v float64, beta ...float64) float64 {
	return v * (1.0 / (1 + math.Exp(beta[0]*-v)))
}

func swishBDeriv(_, _ int, v float64, beta ...float64) float64 {
	prod := v * beta[0]
	exp := math.Exp(prod)
	return exp * (exp + prod + 1) / ((exp + 1) * (exp + 1))
}

func swishBBetaDeriv(v, beta float64) float64 {
	prod := v * beta
	exp := math.Exp(-prod)
	return (v * v * exp) / ((exp + 1) * (exp + 1))
}

// Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function" by Diganta Misra, 2019.
// (https://arxiv.org/pdf/1908.08681.pdf)
func mish(_, _ int, v float64) float64 {
	return v * math.Tanh(math.Log(1+math.Exp(v)))
}

func mishDeriv(_, _ int, v float64) float64 {
	exp := math.Exp(v)
	exp2 := math.Exp(2 * v)
	exp3 := math.Exp(3 * v)
	omega := 4.0*(v+1.0) + 4.0*exp2 + exp3 + exp*(4.0*v+6.0)
	delta := 2*exp + exp2 + 2.0
	return exp * (omega / (delta * delta))
}

func gelu(_, _ int, v float64) float64 {
	return 0.5 * v * (1.0 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*math.Pow(v, 3.0))))
}

func geluDeriv(_, _ int, x float64) float64 {
	x3 := math.Pow(x, 3)
	return 0.5*math.Tanh(0.0356774*x3+0.797885*x) +
		(0.0535161*x3+0.398942*x)*
			math.Pow(1.0/math.Cosh(0.0356774*x3+0.797885*x), 2) + 0.5
}
