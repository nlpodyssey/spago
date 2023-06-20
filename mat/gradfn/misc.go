// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/nlpodyssey/spago/mat"
	"math"
)

// Tan is an operator to perform element-wise tangent.
type Tan[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewTan returns a new UnaryElementwise tangent function.
func NewTan[O mat.Tensor](x O) *Tan[O] {
	return &Tan[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  tan,
			df: tanDeriv,
		},
	}
}

// Tanh is an operator to perform element-wise hyperbolic tangent.
type Tanh[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewTanh returns a new UnaryElementwise hyperbolic tangent function.
func NewTanh[O mat.Tensor](x O) *Tanh[O] {
	return &Tanh[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  tanh,
			df: tanhDeriv,
		},
	}
}

// HardSigmoid is an operator to perform element-wise hard sigmoid.
type HardSigmoid[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewHardSigmoid returns a new UnaryElementwise hard sigmoid function.
func NewHardSigmoid[O mat.Tensor](x O) *HardSigmoid[O] {
	return &HardSigmoid[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  hardSigmoid,
			df: hardSigmoidDeriv,
		},
	}
}

// HardTanh is an operator to perform element-wise hard hyperbolic tangent.
type HardTanh[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewHardTanh returns a new UnaryElementwise hard hyperbolic tangent function.
func NewHardTanh[O mat.Tensor](x O) *HardTanh[O] {
	return &HardTanh[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  hardTanh,
			df: hardTanhDeriv,
		},
	}
}

// ReLU is an operator to perform element-wise Rectified Linear Unit (ReLU)
type ReLU[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewReLU returns a new UnaryElementwise Rectified Linear Unit (ReLU) function.
func NewReLU[O mat.Tensor](x O) *ReLU[O] {
	return &ReLU[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  relu,
			df: reluDeriv,
		},
	}
}

// Softsign is an operator to perform element-wise softsign.
type Softsign[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewSoftsign returns a new UnaryElementwise softsign function.
func NewSoftsign[O mat.Tensor](x O) *Softsign[O] {
	return &Softsign[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  softsign,
			df: softsignDeriv,
		},
	}
}

// Cos is an operator to perform element-wise cos.
type Cos[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewCos returns a new UnaryElementwise cos function.
func NewCos[O mat.Tensor](x O) *Cos[O] {
	return &Cos[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  func(_, _ int, v float64) float64 { return math.Cos(v) },
			df: func(_, _ int, v float64) float64 { return -math.Sin(v) },
		},
	}
}

// Sin is an operator to perform element-wise sin.
type Sin[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewSin returns a new UnaryElementwise sine function.
func NewSin[O mat.Tensor](x O) *Sin[O] {
	return &Sin[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  func(i, j int, v float64) float64 { return math.Sin(v) },
			df: func(i, j int, v float64) float64 { return math.Cos(v) },
		},
	}
}

// Neg is an operator to perform element-wise f(x) = -x
type Neg[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewNeg returns a new UnaryElementwise f(x) = -x function.
func NewNeg[O mat.Tensor](x O) *Neg[O] {
	return &Neg[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  func(i, j int, v float64) float64 { return -v },
			df: func(i, j int, v float64) float64 { return -1.0 },
		},
	}
}

// Reciprocal is an operator to perform element-wise reciprocal.
type Reciprocal[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewReciprocal returns a new UnaryElementwise reciprocal function.
func NewReciprocal[O mat.Tensor](x O) *Reciprocal[O] {
	return &Reciprocal[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  func(i, j int, v float64) float64 { return 1.0 / v },
			df: func(i, j int, v float64) float64 { return -1.0 / (v * v) },
		},
	}
}

// Abs is an operator to perform element-wise absolute value function.
type Abs[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewAbs returns a new UnaryElementwise absolute value function.
func NewAbs[O mat.Tensor](x O) *Abs[O] {
	return &Abs[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  func(i, j int, v float64) float64 { return math.Abs(v) },
			df: absDeriv,
		},
	}
}

// Mish is an operator to perform element-wise mish.
type Mish[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewMish returns a new UnaryElementwise Mish function.
//
// Mish is a self-regularized non-monotonic activation function which can be
// mathematically defined as f(x) = x * tanh(softplus(x)).
//
// Reference: "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
// by Diganta Misra, 2019 (https://arxiv.org/pdf/1908.08681.pdf)
func NewMish[O mat.Tensor](x O) *Mish[O] {
	return &Mish[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  mish,
			df: mishDeriv,
		},
	}
}

// GELU is an operator to perform element-wise GELU.
type GELU[O mat.Tensor] struct {
	*UnaryElementwise[O]
}

// NewGELU returns a new UnaryElementwise Gaussian Error Linear Unit (GELU) function.
func NewGELU[O mat.Tensor](x O) *GELU[O] {
	return &GELU[O]{
		UnaryElementwise: &UnaryElementwise[O]{
			x:  x,
			f:  gelu,
			df: geluDeriv,
		},
	}
}

// NewSiLU (Sigmoid Linear Unit) returns a new function of the form f(x) = x * sigmoid(x).
// The function in an alias of NewSwish.
func NewSiLU[O mat.Tensor](x O) *Swish[O] {
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
