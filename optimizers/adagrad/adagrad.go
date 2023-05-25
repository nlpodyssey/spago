// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adagrad

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/optimizers"
)

var _ optimizers.StrategyConfig = &Config{}

// Config provides configuration settings for an AdaGrad optimizer.
type Config struct {
	optimizers.StrategyConfig
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

var _ optimizers.Strategy = &AdaGrad[float32]{}

// AdaGrad assigns a different learning rate to each parameter using the sum of squares of its all historical gradients.
// References
//
//	Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
//	http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
type AdaGrad[T float.DType] struct {
	Config
}

// New returns a new AdaGrad optimizer, initialized according to the given configuration.
func New[T float.DType](c Config) *AdaGrad[T] {
	return &AdaGrad[T]{Config: c}
}

const m = 0

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *AdaGrad[_]) Label() int {
	return optimizers.AdaGrad
}

func (o *AdaGrad[T]) NewState(shape ...int) any {
	r, c := shape[0], shape[1]
	return []mat.Matrix{mat.NewEmptyDense[T](r, c)} // m at index 0
}

// CalcDelta returns the difference between the current params and where the method wants it to be.
func (o *AdaGrad[T]) CalcDelta(param *nn.Param) mat.Matrix {
	grads := param.Grad()
	supp := param.GetOrSetState(o.NewState).([]mat.Matrix)
	return o.calcDelta(grads, supp)
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
