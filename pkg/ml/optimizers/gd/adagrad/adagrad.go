// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adagrad

import (
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/optimizers/gd"
)

type Config struct {
	LR      float64
	Epsilon float64
}

func NewConfig(lr, epsilon float64) Config {
	return Config{
		LR:      lr,
		Epsilon: epsilon,
	}
}

func NewDefaultConfig() Config {
	return Config{
		LR:      0.01,
		Epsilon: 1.0e-8,
	}
}

var _ gd.OptimizationMethod = &AdaGrad{}

// AdaGrad assigns a different learning rate to each parameter using the sum of squares of its all historical gradients.
// References
//     Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
//     http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
type AdaGrad struct {
	Config
}

func New(c Config) *AdaGrad {
	return &AdaGrad{Config: c}
}

const m = 0

func (o *AdaGrad) Name() gd.MethodName {
	return gd.AdaGrad
}

func (o *AdaGrad) NewSupport(r, c int) *gd.Support {
	return &gd.Support{Name: o.Name(), Data: []mat.Matrix{mat.NewEmptyDense(r, c)}} // m at index 0
}

func (o *AdaGrad) Delta(param gd.Optimizable) mat.Matrix {
	return o.calcDelta(param.Grad(), param.GetOrSetSupport(o).Data)
}

// m = m + grads*grads
// delta = (grads / (sqrt(m) + eps)) * lr
func (o *AdaGrad) calcDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[m].AddInPlace(grads.Prod(grads))
	buf := mat.Sqrt(supp[m])
	buf.AddScalarInPlace(o.Epsilon)
	delta := grads.Div(buf)
	delta.ProdScalarInPlace(o.LR)
	return delta
}
