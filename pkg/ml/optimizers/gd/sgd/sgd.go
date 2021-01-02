// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgd

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
)

var _ gd.MethodConfig = &Config{}

// Config provides configuration settings for an SGD optimizer.
type Config struct {
	gd.MethodConfig
	LR       mat.Float
	Mu       mat.Float
	Nesterov bool
}

// NewConfig returns a new SGD Config.
func NewConfig(lr, momentum mat.Float, nesterov bool) Config {
	return Config{
		LR:       lr,
		Mu:       momentum,
		Nesterov: nesterov,
	}
}

var _ gd.Method = &SGD{}

// SGD implements the SGD gradient descent optimization method.
type SGD struct {
	Config
	Alpha mat.Float
}

// New returns a new SGD optimizer, initialized according to the given configuration.
func New(c Config) *SGD {
	return &SGD{Config: c, Alpha: c.LR}
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *SGD) Label() int {
	return gd.SGD
}

const (
	v     int = 0
	buf   int = 1
	vPrev int = 2
	vTmp  int = 3
)

// NewSupport returns a new support structure with the given dimensions.
func (o *SGD) NewSupport(r, c int) *nn.Payload {
	if o.Mu == 0.0 {
		// Vanilla SGD doesn't require any support structure, this is just to avoid memory allocation
		return &nn.Payload{
			Label: o.Label(),
			Data:  []mat.Matrix{mat.NewEmptyDense(r, c)}, // v at index 0
		}
	}
	if !o.Nesterov {
		supp := make([]mat.Matrix, 2, 2)
		supp[v] = mat.NewEmptyDense(r, c)
		supp[buf] = mat.NewEmptyDense(r, c)
		return &nn.Payload{
			Label: o.Label(),
			Data:  supp,
		}
	}
	supp := make([]mat.Matrix, 4, 4)
	supp[v] = mat.NewEmptyDense(r, c)
	supp[buf] = mat.NewEmptyDense(r, c)
	supp[vPrev] = mat.NewEmptyDense(r, c)
	supp[vTmp] = mat.NewEmptyDense(r, c)
	return &nn.Payload{
		Label: o.Label(),
		Data:  supp,
	}
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *SGD) Delta(param nn.Param) mat.Matrix {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload(param, o).Data)
}

func (o *SGD) calcDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	if o.Mu == 0.0 {
		return o.calcVanillaSGD(grads, supp)
	} else if o.Nesterov {
		return o.calcNesterovMomentumDelta(grads, supp)
	} else {
		return o.calcMomentumDelta(grads, supp)
	}
}

func (o *SGD) calcVanillaSGD(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[v].ProdMatrixScalarInPlace(grads, o.Alpha)
	return supp[v]
}

func (o *SGD) calcMomentumDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[buf].ProdMatrixScalarInPlace(grads, o.Alpha)
	supp[v].ProdScalarInPlace(o.Mu)
	supp[v].AddInPlace(supp[buf])
	return supp[v]
}

func (o *SGD) calcNesterovMomentumDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[buf].ProdMatrixScalarInPlace(grads, o.Alpha)
	supp[vPrev].ProdMatrixScalarInPlace(supp[v], o.Mu)
	supp[v].ProdScalarInPlace(o.Mu)
	supp[v].AddInPlace(supp[buf]) // += grad * alpha
	supp[vTmp].ProdMatrixScalarInPlace(supp[v], 1.0+o.Mu)
	supp[vTmp].SubInPlace(supp[vPrev])
	return supp[vTmp]
}
