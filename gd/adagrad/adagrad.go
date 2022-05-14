// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adagrad

import (
	"github.com/nlpodyssey/spago/gd"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ gd.MethodConfig = &Config{}

// Config provides configuration settings for an AdaGrad optimizer.
type Config struct {
	gd.MethodConfig
	LR      float64
	Epsilon float64
}

// NewConfig returns a new AdaGrad Config.
func NewConfig(lr, epsilon float64) Config {
	return Config{
		LR:      lr,
		Epsilon: epsilon,
	}
}

// NewDefaultConfig returns a new Config with generically reasonable default values.
func NewDefaultConfig() Config {
	return Config{
		LR:      0.01,
		Epsilon: 1.0e-8,
	}
}

var _ gd.Method[float32] = &AdaGrad[float32]{}

// AdaGrad assigns a different learning rate to each parameter using the sum of squares of its all historical gradients.
// References
//     Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
//     http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
type AdaGrad[T mat.DType] struct {
	Config
}

// New returns a new AdaGrad optimizer, initialized according to the given configuration.
func New[T mat.DType](c Config) *AdaGrad[T] {
	return &AdaGrad[T]{Config: c}
}

const m = 0

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *AdaGrad[_]) Label() int {
	return gd.AdaGrad
}

// NewSupport returns a new support structure with the given dimensions.
func (o *AdaGrad[T]) NewSupport(r, c int) *nn.Payload[T] {
	return &nn.Payload[T]{
		Label: o.Label(),
		Data:  []mat.Matrix{mat.NewEmptyDense[T](r, c)}, // m at index 0
	}
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *AdaGrad[T]) Delta(param nn.Param[T]) mat.Matrix {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload[T](param, o).Data)
}

// m = m + grads*grads
// delta = (grads / (sqrt(m) + eps)) * lr
func (o *AdaGrad[T]) calcDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[m].AddInPlace(grads.Prod(grads))
	buf := supp[m].Sqrt() // TODO: this was "buf := mat.SqrtMatrix(supp[m])", is it the same?
	buf.AddScalarInPlace(o.Epsilon)
	delta := grads.Div(buf)
	delta.ProdScalarInPlace(o.LR)
	return delta
}
