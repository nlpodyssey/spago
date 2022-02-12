// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Tan is an operator to perform element-wise tangent.
type Tan[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewTan returns a new UnaryElementwise tangent function.
func NewTan[T mat.DType](x Operand[T]) *Tan[T] {
	return &Tan[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  tan[T],
			df: tanDeriv[T],
		},
	}
}

// Tanh is an operator to perform element-wise hyperbolic tangent.
type Tanh[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewTanh returns a new UnaryElementwise hyperbolic tangent function.
func NewTanh[T mat.DType](x Operand[T]) *Tanh[T] {
	return &Tanh[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  tanh[T],
			df: tanhDeriv[T],
		},
	}
}

// Sigmoid is an operator to perform element-wise sigmoid.
type Sigmoid[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewSigmoid returns a new UnaryElementwise sigmoid function.
func NewSigmoid[T mat.DType](x Operand[T]) *Sigmoid[T] {
	return &Sigmoid[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  sigmoid[T],
			df: sigmoidDeriv[T],
		},
	}
}

// HardSigmoid is an operator to perform element-wise hard sigmoid.
type HardSigmoid[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewHardSigmoid returns a new UnaryElementwise hard sigmoid function.
func NewHardSigmoid[T mat.DType](x Operand[T]) *HardSigmoid[T] {
	return &HardSigmoid[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  hardSigmoid[T],
			df: hardSigmoidDeriv[T],
		},
	}
}

// HardTanh is an operator to perform element-wise hard hyperbolic tangent.
type HardTanh[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewHardTanh returns a new UnaryElementwise hard hyperbolic tangent function.
func NewHardTanh[T mat.DType](x Operand[T]) *HardTanh[T] {
	return &HardTanh[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  hardTanh[T],
			df: hardTanhDeriv[T],
		},
	}
}

// ReLU is an operator to perform element-wise Rectified Linear Unit (ReLU)
type ReLU[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewReLU returns a new UnaryElementwise Rectified Linear Unit (ReLU) function.
func NewReLU[T mat.DType](x Operand[T]) *ReLU[T] {
	return &ReLU[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  relu[T],
			df: reluDeriv[T],
		},
	}
}

// Softsign is an operator to perform element-wise softsign.
type Softsign[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewSoftsign returns a new UnaryElementwise softsign function.
func NewSoftsign[T mat.DType](x Operand[T]) *Softsign[T] {
	return &Softsign[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  softsign[T],
			df: softsignDeriv[T],
		},
	}
}

// Cos is an operator to perform element-wise cos.
type Cos[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewCos returns a new UnaryElementwise cos function.
func NewCos[T mat.DType](x Operand[T]) *Cos[T] {
	return &Cos[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  func(i, j int, v T) T { return mat.Cos(v) },
			df: func(i, j int, v T) T { return -mat.Sin(v) },
		},
	}
}

// Sin is an operator to perform element-wise sin.
type Sin[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewSin returns a new UnaryElementwise sine function.
func NewSin[T mat.DType](x Operand[T]) *Sin[T] {
	return &Sin[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  func(i, j int, v T) T { return mat.Sin(v) },
			df: func(i, j int, v T) T { return mat.Cos(v) },
		},
	}
}

// Exp is an operator to perform element-wise base-e exponential.
type Exp[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewExp returns a new UnaryElementwise base-e exponential function.
func NewExp[T mat.DType](x Operand[T]) *Exp[T] {
	return &Exp[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  func(i, j int, v T) T { return mat.Exp(v) },
			df: func(i, j int, v T) T { return mat.Exp(v) },
		},
	}
}

// Log is an operator to perform element-wise natural logarithm.
type Log[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewLog returns a new UnaryElementwise natural logarithm function.
func NewLog[T mat.DType](x Operand[T]) *Log[T] {
	return &Log[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  safeLog[T],
			df: safeLogDeriv[T],
		},
	}
}

// Neg is an operator to perform element-wise f(x) = -x
type Neg[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewNeg returns a new UnaryElementwise f(x) = -x function.
func NewNeg[T mat.DType](x Operand[T]) *Neg[T] {
	return &Neg[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  func(i, j int, v T) T { return -v },
			df: func(i, j int, v T) T { return -1.0 },
		},
	}
}

// Reciprocal is an operator to perform element-wise reciprocal.
type Reciprocal[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewReciprocal returns a new UnaryElementwise reciprocal function.
func NewReciprocal[T mat.DType](x Operand[T]) *Reciprocal[T] {
	return &Reciprocal[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  func(i, j int, v T) T { return 1.0 / v },
			df: func(i, j int, v T) T { return -1.0 / (v * v) },
		},
	}
}

// Abs is an operator to perform element-wise absolute value function.
type Abs[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewAbs returns a new UnaryElementwise absolute value function.
func NewAbs[T mat.DType](x Operand[T]) *Abs[T] {
	return &Abs[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  func(i, j int, v T) T { return mat.Abs(v) },
			df: absDeriv[T],
		},
	}
}

// Mish is an operator to perform element-wise mish.
type Mish[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewMish returns a new UnaryElementwise Mish function.
//
// Mish is a self-regularized non-monotonic activation function which can be
// mathematically defined as f(x) = x * tanh(softplus(x)).
//
// Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
// by Diganta Misra, 2019 (https://arxiv.org/pdf/1908.08681.pdf)
func NewMish[T mat.DType](x Operand[T]) *Mish[T] {
	return &Mish[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  mish[T],
			df: mishDeriv[T],
		},
	}
}

// GELU is an operator to perform element-wise GELU.
type GELU[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewGELU returns a new UnaryElementwise Gaussian Error Linear Unit (GELU) function.
func NewGELU[T mat.DType](x Operand[T]) *GELU[T] {
	return &GELU[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  gelu[T],
			df: geluDeriv[T],
		},
	}
}

// Sqrt is an operator to perform element-wise square root.
type Sqrt[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewSqrt returns a new UnaryElementwise square root function.
func NewSqrt[T mat.DType](x Operand[T]) *Sqrt[T] {
	return &Sqrt[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  func(i, j int, v T) T { return mat.Sqrt(v) },
			df: func(i, j int, v T) T { return 0.5 * mat.Pow(v, -0.5) },
		},
	}
}

// Swish is an operator to perform element-wise x * sigmoid(x).
type Swish[T mat.DType] struct {
	*UnaryElementwise[T]
}

// NewSwish returns a new function of the form f(x) = x * sigmoid(x).
func NewSwish[T mat.DType](x Operand[T]) *Swish[T] {
	return &Swish[T]{
		UnaryElementwise: &UnaryElementwise[T]{
			x:  x,
			f:  swish[T],
			df: swishDeriv[T],
		},
	}
}

// NewSiLU (Sigmoid Linear Unit) returns a new function of the form f(x) = x * sigmoid(x).
// The function in an alias of NewSwish.
func NewSiLU[T mat.DType](x Operand[T]) *Swish[T] {
	return NewSwish(x)
}

func absDeriv[T mat.DType](_, _ int, v T) T {
	if v < 0 {
		return -1
	} else if v > 0 {
		return 1
	} else {
		return 0 // undefined
	}
}

// safeLog is a simple work-around that make the math.Log() safe for zero or negative values
func safeLog[T mat.DType](_, _ int, v T) T {
	if v > 0.0 {
		return mat.Log(v)
	}
	if v == 0.0 {
		return mat.Log[T](1.0e-08)
	}
	panic("ag: invalid log for negative values")
}

func safeLogDeriv[T mat.DType](_, _ int, v T) T {
	if v > 0.0 {
		return 1.0 / v
	} else if v == 0.0 {
		return 1.0 / 1.0e-08
	} else {
		panic("ag: invalid log for negative values")
	}
}

func tan[T mat.DType](_, _ int, v T) T {
	return mat.Tan(v)
}

func tanDeriv[T mat.DType](i, j int, v T) T {
	return 1.0 / square(i, j, mat.Cos(v))
}

func square[T mat.DType](_, _ int, v T) T {
	return v * v
}

func tanh[T mat.DType](_, _ int, v T) T {
	return mat.Tanh(v)
}

func tanhDeriv[T mat.DType](_, _ int, v T) T {
	return 1.0 - mat.Pow(mat.Tanh(v), 2.0)
}

func sigmoid[T mat.DType](_, _ int, v T) T {
	return 1.0 / (1 + mat.Exp(-v))
}

func sigmoidDeriv[T mat.DType](i, j int, v T) T {
	fx := sigmoid(i, j, v)
	return fx * (1.0 - fx)
}

func hardSigmoid[T mat.DType](_, _ int, v T) T {
	if v > 2.5 {
		return 1.0
	} else if v < -2.5 {
		return 0.0
	} else {
		return 0.2*v + 0.5
	}
}

func hardSigmoidDeriv[T mat.DType](_, _ int, v T) T {
	if v < 2.5 && v > -2.5 {
		return 0.2
	}
	return 0.0
}

func hardTanh[T mat.DType](_, _ int, v T) T {
	if v > 1.0 {
		return 1.0
	} else if v < -1.0 {
		return -1.0
	} else {
		return v
	}
}

func hardTanhDeriv[T mat.DType](_, _ int, v T) T {
	if v < 1.0 && v > -1.0 {
		return 1.0
	}
	return 0.0
}

func relu[T mat.DType](_, _ int, v T) T {
	return mat.Max(0.0, v)
}

func reluDeriv[T mat.DType](_, _ int, v T) T {
	if v >= 0.0 {
		return 1.0
	}
	return 0.0
}

func softsign[T mat.DType](_, _ int, v T) T {
	return v / (1.0 + mat.Abs(v))
}

func softsignDeriv[T mat.DType](i, j int, v T) T {
	return mat.Pow(1.0-mat.Abs(softsign(i, j, v)), 2.0)
}

func celu[T mat.DType](_, _ int, v T, alpha ...T) T {
	if v <= 0 {
		return alpha[0] * (mat.Exp(v/alpha[0]) - 1)
	} else if v > 0 {
		return v
	}
	return 0
}

func celuDeriv[T mat.DType](_, _ int, v T, alpha ...T) T {
	if v <= 0 {
		return mat.Exp(v / alpha[0])
	} else if v > 0 {
		return 1
	}
	return 0
}

func elu[T mat.DType](_, _ int, v T, alpha ...T) T {
	if v <= 0 {
		return alpha[0] * (mat.Exp(v) - 1)
	} else if v > 0 {
		return v
	}
	return 0
}

func eluDeriv[T mat.DType](_, _ int, v T, alpha ...T) T {
	if v <= 0 {
		return alpha[0] * mat.Exp(v)
	} else if v > 0 {
		return 1
	}
	return 0
}

func leakyReLU[T mat.DType](_, _ int, v T, alpha ...T) T {
	if v <= 0 {
		return alpha[0] * v // slope * v
	} else if v > 0 {
		return v
	}
	return 0
}

func leakyReLUDeriv[T mat.DType](_, _ int, v T, alpha ...T) T {
	if v <= 0 {
		return alpha[0] // slope
	} else if v > 0 {
		return 1
	}
	return 0
}

// alpha[0] is the alpha
// alpha[1] is the scale
func selu[T mat.DType](_, _ int, v T, alpha ...T) T {
	scale := alpha[1]
	if v <= 0 {
		return scale * alpha[0] * (mat.Exp(v) - 1)
	} else if v > 0 {
		return scale * v
	}
	return 0
}

// alpha[0] is the alpha
// alpha[1] is the scale
func seluDeriv[T mat.DType](_, _ int, v T, alpha ...T) T {
	scale := alpha[1]
	if v <= 0 {
		return scale * alpha[0] * mat.Exp(v)
	} else if v > 0 {
		return scale
	}
	return 0
}

func softPlus[T mat.DType](_, _ int, v T, alpha ...T) T {
	threshold := alpha[1]
	beta := alpha[0]
	if v <= threshold {
		return (1 / beta) * mat.Log(1+mat.Exp(beta*v))
	} else if v > threshold {
		return v
	}
	return 0
}

func softPlusDeriv[T mat.DType](_, _ int, v T, alpha ...T) T {
	threshold := alpha[1]
	beta := alpha[0]
	if v <= threshold {
		exp := mat.Exp(v * beta)
		return exp / (exp + 1)
	} else if v > threshold {
		return 1
	}
	return 0
}

func softShrink[T mat.DType](_, _ int, v T, alpha ...T) T {
	lambda := alpha[0]
	if v < -lambda {
		return v + lambda
	} else if v > lambda {
		return v - lambda
	}
	return 0
}

func softShrinkDeriv[T mat.DType](_, _ int, v T, alpha ...T) T {
	lambda := alpha[0]
	if v < -lambda {
		return 1
	} else if v > lambda {
		return 1
	}
	return 0
}

func threshold[T mat.DType](_, _ int, v T, alpha ...T) T {
	value := alpha[1]
	threshold := alpha[0]
	if v <= threshold {
		return value
	} else if v > threshold {
		return v
	}
	return 0
}

func thresholdDeriv[T mat.DType](_, _ int, v T, alpha ...T) T {
	threshold := alpha[0]
	if v <= threshold {
		return 0
	} else if v > threshold {
		return 1
	}
	return 0
}

func swish[T mat.DType](_, _ int, v T) T {
	return v * (1.0 / (1 + mat.Exp(-v)))
}

func swishDeriv[T mat.DType](i, j int, v T) T {
	return swishBDeriv(i, j, v, 1.0)
}

func swishB[T mat.DType](_, _ int, v T, beta ...T) T {
	return v * (1.0 / (1 + mat.Exp(beta[0]*-v)))
}

func swishBDeriv[T mat.DType](_, _ int, v T, beta ...T) T {
	prod := v * beta[0]
	exp := mat.Exp(prod)
	return exp * (exp + prod + 1) / ((exp + 1) * (exp + 1))
}

func swishBBetaDeriv[T mat.DType](v T, beta T) T {
	prod := v * beta
	exp := mat.Exp(-prod)
	return (v * v * exp) / ((exp + 1) * (exp + 1))
}

// Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function" by Diganta Misra, 2019.
// (https://arxiv.org/pdf/1908.08681.pdf)
func mish[T mat.DType](_, _ int, v T) T {
	return v * mat.Tanh(mat.Log(1+mat.Exp(v)))
}

func mishDeriv[T mat.DType](_, _ int, v T) T {
	exp := mat.Exp(v)
	exp2 := mat.Exp(2 * v)
	exp3 := mat.Exp(3 * v)
	omega := 4.0*(v+1.0) + 4.0*exp2 + exp3 + exp*(4.0*v+6.0)
	delta := 2*exp + exp2 + 2.0
	return exp * (omega / (delta * delta))
}

func gelu[T mat.DType](_, _ int, v T) T {
	return 0.5 * v * (1.0 + mat.Tanh(mat.Sqrt(2/mat.Pi[T]())*(v+0.044715*mat.Pow(v, 3.0))))
}

func geluDeriv[T mat.DType](_, _ int, x T) T {
	x3 := mat.Pow(x, 3)
	return 0.5*mat.Tanh(0.0356774*x3+0.797885*x) +
		(0.0535161*x3+0.398942*x)*
			mat.Pow(1.0/mat.Cosh(0.0356774*x3+0.797885*x), 2) + 0.5
}
