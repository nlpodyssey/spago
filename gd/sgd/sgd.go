// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgd

import (
	"github.com/nlpodyssey/spago/gd"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ gd.MethodConfig = &Config{}

// Config provides configuration settings for an SGD optimizer.
type Config struct {
	gd.MethodConfig
	LR       float64
	Mu       float64
	Nesterov bool
}

// NewConfig returns a new SGD Config.
func NewConfig(lr, momentum float64, nesterov bool) Config {
	return Config{
		LR:       lr,
		Mu:       momentum,
		Nesterov: nesterov,
	}
}

var _ gd.Method[float32] = &SGD[float32]{}

// SGD implements the SGD gradient descent optimization method.
type SGD[T mat.DType] struct {
	Config
	Alpha float64
}

// New returns a new SGD optimizer, initialized according to the given configuration.
func New[T mat.DType](c Config) *SGD[T] {
	return &SGD[T]{Config: c, Alpha: c.LR}
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *SGD[_]) Label() int {
	return gd.SGD
}

const (
	v     int = 0
	buf   int = 1
	vPrev int = 2
	vTmp  int = 3
)

// NewSupport returns a new support structure with the given dimensions.
func (o *SGD[T]) NewSupport(r, c int) *nn.Payload[T] {
	if o.Mu == 0.0 {
		// Vanilla SGD doesn't require any support structure, this is just to avoid memory allocation
		return &nn.Payload[T]{
			Label: o.Label(),
			Data:  []mat.Matrix[T]{mat.NewEmptyDense[T](r, c)}, // v at index 0
		}
	}
	if !o.Nesterov {
		supp := make([]mat.Matrix[T], 2)
		supp[v] = mat.NewEmptyDense[T](r, c)
		supp[buf] = mat.NewEmptyDense[T](r, c)
		return &nn.Payload[T]{
			Label: o.Label(),
			Data:  supp,
		}
	}
	supp := make([]mat.Matrix[T], 4)
	supp[v] = mat.NewEmptyDense[T](r, c)
	supp[buf] = mat.NewEmptyDense[T](r, c)
	supp[vPrev] = mat.NewEmptyDense[T](r, c)
	supp[vTmp] = mat.NewEmptyDense[T](r, c)
	return &nn.Payload[T]{
		Label: o.Label(),
		Data:  supp,
	}
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *SGD[T]) Delta(param nn.Param[T]) mat.Matrix[T] {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload[T](param, o).Data)
}

func (o *SGD[T]) calcDelta(grads mat.Matrix[T], supp []mat.Matrix[T]) mat.Matrix[T] {
	if o.Mu == 0.0 {
		return o.calcVanillaSGD(grads, supp)
	} else if o.Nesterov {
		return o.calcNesterovMomentumDelta(grads, supp)
	} else {
		return o.calcMomentumDelta(grads, supp)
	}
}

func (o *SGD[T]) calcVanillaSGD(grads mat.Matrix[T], supp []mat.Matrix[T]) mat.Matrix[T] {
	supp[v].ProdMatrixScalarInPlace(grads, o.Alpha)
	return supp[v]
}

func (o *SGD[T]) calcMomentumDelta(grads mat.Matrix[T], supp []mat.Matrix[T]) mat.Matrix[T] {
	supp[buf].ProdMatrixScalarInPlace(grads, o.Alpha)
	supp[v].ProdScalarInPlace(o.Mu)
	supp[v].AddInPlace(supp[buf])
	return supp[v]
}

func (o *SGD[T]) calcNesterovMomentumDelta(grads mat.Matrix[T], supp []mat.Matrix[T]) mat.Matrix[T] {
	supp[buf].ProdMatrixScalarInPlace(grads, o.Alpha)
	supp[vPrev].ProdMatrixScalarInPlace(supp[v], o.Mu)
	supp[v].ProdScalarInPlace(o.Mu)
	supp[v].AddInPlace(supp[buf]) // += grad * alpha
	supp[vTmp].ProdMatrixScalarInPlace(supp[v], 1.0+o.Mu)
	supp[vTmp].SubInPlace(supp[vPrev])
	return supp[vTmp]
}
