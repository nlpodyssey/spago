// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"math"

	"github.com/nlpodyssey/spago/mat"
)

// Tan is an operator to perform element-wise tangent.
type Tan[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewTan returns a new UnaryElementwise tangent function.
func NewTan[T mat.DType, O Operand[T]](x O) *Tan[T, O] {
	return &Tan[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        tan,
			df:       tanDeriv,
			operands: []O{x},
		},
	}
}

// Tanh is an operator to perform element-wise hyperbolic tangent.
type Tanh[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewTanh returns a new UnaryElementwise hyperbolic tangent function.
func NewTanh[T mat.DType, O Operand[T]](x O) *Tanh[T, O] {
	return &Tanh[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        tanh,
			df:       tanhDeriv,
			operands: []O{x},
		},
	}
}

// Sigmoid is an operator to perform element-wise sigmoid.
type Sigmoid[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewSigmoid returns a new UnaryElementwise sigmoid function.
func NewSigmoid[T mat.DType, O Operand[T]](x O) *Sigmoid[T, O] {
	return &Sigmoid[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        sigmoid,
			df:       sigmoidDeriv,
			operands: []O{x},
		},
	}
}

// HardSigmoid is an operator to perform element-wise hard sigmoid.
type HardSigmoid[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewHardSigmoid returns a new UnaryElementwise hard sigmoid function.
func NewHardSigmoid[T mat.DType, O Operand[T]](x O) *HardSigmoid[T, O] {
	return &HardSigmoid[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        hardSigmoid,
			df:       hardSigmoidDeriv,
			operands: []O{x},
		},
	}
}

// HardTanh is an operator to perform element-wise hard hyperbolic tangent.
type HardTanh[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewHardTanh returns a new UnaryElementwise hard hyperbolic tangent function.
func NewHardTanh[T mat.DType, O Operand[T]](x O) *HardTanh[T, O] {
	return &HardTanh[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        hardTanh,
			df:       hardTanhDeriv,
			operands: []O{x},
		},
	}
}

// ReLU is an operator to perform element-wise Rectified Linear Unit (ReLU)
type ReLU[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewReLU returns a new UnaryElementwise Rectified Linear Unit (ReLU) function.
func NewReLU[T mat.DType, O Operand[T]](x O) *ReLU[T, O] {
	return &ReLU[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        relu,
			df:       reluDeriv,
			operands: []O{x},
		},
	}
}

// Softsign is an operator to perform element-wise softsign.
type Softsign[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewSoftsign returns a new UnaryElementwise softsign function.
func NewSoftsign[T mat.DType, O Operand[T]](x O) *Softsign[T, O] {
	return &Softsign[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        softsign,
			df:       softsignDeriv,
			operands: []O{x},
		},
	}
}

// Cos is an operator to perform element-wise cos.
type Cos[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewCos returns a new UnaryElementwise cos function.
func NewCos[T mat.DType, O Operand[T]](x O) *Cos[T, O] {
	return &Cos[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        func(_, _ int, v float64) float64 { return math.Cos(v) },
			df:       func(_, _ int, v float64) float64 { return -math.Sin(v) },
			operands: []O{x},
		},
	}
}

// Sin is an operator to perform element-wise sin.
type Sin[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewSin returns a new UnaryElementwise sine function.
func NewSin[T mat.DType, O Operand[T]](x O) *Sin[T, O] {
	return &Sin[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Sin(v) },
			df:       func(i, j int, v float64) float64 { return math.Cos(v) },
			operands: []O{x},
		},
	}
}

// Exp is an operator to perform element-wise base-e exponential.
type Exp[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewExp returns a new UnaryElementwise base-e exponential function.
func NewExp[T mat.DType, O Operand[T]](x O) *Exp[T, O] {
	return &Exp[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Exp(v) },
			df:       func(i, j int, v float64) float64 { return math.Exp(v) },
			operands: []O{x},
		},
	}
}

// Log is an operator to perform element-wise natural logarithm.
type Log[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewLog returns a new UnaryElementwise natural logarithm function.
func NewLog[T mat.DType, O Operand[T]](x O) *Log[T, O] {
	return &Log[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        safeLog,
			df:       safeLogDeriv,
			operands: []O{x},
		},
	}
}

// Neg is an operator to perform element-wise f(x) = -x
type Neg[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewNeg returns a new UnaryElementwise f(x) = -x function.
func NewNeg[T mat.DType, O Operand[T]](x O) *Neg[T, O] {
	return &Neg[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return -v },
			df:       func(i, j int, v float64) float64 { return -1.0 },
			operands: []O{x},
		},
	}
}

// Reciprocal is an operator to perform element-wise reciprocal.
type Reciprocal[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewReciprocal returns a new UnaryElementwise reciprocal function.
func NewReciprocal[T mat.DType, O Operand[T]](x O) *Reciprocal[T, O] {
	return &Reciprocal[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return 1.0 / v },
			df:       func(i, j int, v float64) float64 { return -1.0 / (v * v) },
			operands: []O{x},
		},
	}
}

// Abs is an operator to perform element-wise absolute value function.
type Abs[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewAbs returns a new UnaryElementwise absolute value function.
func NewAbs[T mat.DType, O Operand[T]](x O) *Abs[T, O] {
	return &Abs[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Abs(v) },
			df:       absDeriv,
			operands: []O{x},
		},
	}
}

// Mish is an operator to perform element-wise mish.
type Mish[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewMish returns a new UnaryElementwise Mish function.
//
// Mish is a self-regularized non-monotonic activation function which can be
// mathematically defined as f(x) = x * tanh(softplus(x)).
//
// Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
// by Diganta Misra, 2019 (https://arxiv.org/pdf/1908.08681.pdf)
func NewMish[T mat.DType, O Operand[T]](x O) *Mish[T, O] {
	return &Mish[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        mish,
			df:       mishDeriv,
			operands: []O{x},
		},
	}
}

// GELU is an operator to perform element-wise GELU.
type GELU[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewGELU returns a new UnaryElementwise Gaussian Error Linear Unit (GELU) function.
func NewGELU[T mat.DType, O Operand[T]](x O) *GELU[T, O] {
	return &GELU[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        gelu,
			df:       geluDeriv,
			operands: []O{x},
		},
	}
}

// Sqrt is an operator to perform element-wise square root.
type Sqrt[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewSqrt returns a new UnaryElementwise square root function.
func NewSqrt[T mat.DType, O Operand[T]](x O) *Sqrt[T, O] {
	return &Sqrt[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        func(i, j int, v float64) float64 { return math.Sqrt(v) },
			df:       func(i, j int, v float64) float64 { return 0.5 * math.Pow(v, -0.5) },
			operands: []O{x},
		},
	}
}

// Swish is an operator to perform element-wise x * sigmoid(x).
type Swish[T mat.DType, O Operand[T]] struct {
	*UnaryElementwise[T, O]
}

// NewSwish returns a new function of the form f(x) = x * sigmoid(x).
func NewSwish[T mat.DType, O Operand[T]](x O) *Swish[T, O] {
	return &Swish[T, O]{
		UnaryElementwise: &UnaryElementwise[T, O]{
			x:        x,
			f:        swish,
			df:       swishDeriv,
			operands: []O{x},
		},
	}
}

// NewSiLU (Sigmoid Linear Unit) returns a new function of the form f(x) = x * sigmoid(x).
// The function in an alias of NewSwish.
func NewSiLU[T mat.DType, O Operand[T]](x O) *Swish[T, O] {
	return NewSwish[T, O](x)
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

func tanDeriv(i, j int, v float64) float64 {
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

func swishBBetaDeriv[T mat.DType](v T, beta T) T {
	prod := v * beta
	exp := mat.Exp(-prod)
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
