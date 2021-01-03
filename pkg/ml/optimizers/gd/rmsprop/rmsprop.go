// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rmsprop

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
)

var _ gd.MethodConfig = &Config{}

// Config provides configuration settings for an RMSProp optimizer.
type Config struct {
	gd.MethodConfig
	LR      mat.Float
	Epsilon mat.Float
	Decay   mat.Float
}

// NewConfig returns a new RMSProp Config.
func NewConfig(lr, epsilon, decay mat.Float) Config {
	return Config{
		LR:      lr,
		Epsilon: epsilon,
		Decay:   decay,
	}
}

// NewDefaultConfig returns a new Config with generically reasonable default values.
func NewDefaultConfig() Config {
	return Config{
		LR:      0.001,
		Epsilon: 1e-08,
		Decay:   0.95,
	}
}

var _ gd.Method = &RMSProp{}

// The RMSProp method is a variant of AdaGrad where the squared sum of previous gradients is replaced with a moving average.
// References:
//     RMSProp: Divide the gradient by a running average of its recent magnitude
//     http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
type RMSProp struct {
	Config
}

// New returns a new RMSProp optimizer, initialized according to the given configuration.
func New(c Config) *RMSProp {
	return &RMSProp{Config: c}
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *RMSProp) Label() int {
	return gd.RMSProp
}

const v = 0

// NewSupport returns a new support structure with the given dimensions.
func (o *RMSProp) NewSupport(r, c int) *nn.Payload {
	return &nn.Payload{
		Label: gd.RMSProp,
		Data:  []mat.Matrix{mat.NewEmptyDense(r, c)}, // v at index 0
	}
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *RMSProp) Delta(param nn.Param) mat.Matrix {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload(param, o).Data)
}

func (o *RMSProp) calcDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[v].ProdScalarInPlace(o.Decay)
	buf := grads.Prod(grads)
	buf.ProdScalarInPlace(1.0 - o.Decay)
	supp[v].AddInPlace(buf)
	buf2 := mat.SqrtMatrix(supp[v])
	buf2.AddScalarInPlace(o.Epsilon)
	delta := grads.Div(buf2)
	delta.ProdScalarInPlace(o.LR)
	return delta
}
