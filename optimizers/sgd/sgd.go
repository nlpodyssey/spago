// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgd

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/optimizers"
)

var _ optimizers.StrategyConfig = &Config{}

// Config provides configuration settings for an SGD optimizer.
type Config struct {
	optimizers.StrategyConfig
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

var _ optimizers.Strategy = &SGD[float32]{}

// SGD implements the SGD gradient descent optimization method.
type SGD[T float.DType] struct {
	Config
	Alpha float64
}

// New returns a new SGD optimizer, initialized according to the given configuration.
func New[T float.DType](c Config) *SGD[T] {
	return &SGD[T]{Config: c, Alpha: c.LR}
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *SGD[_]) Label() int {
	return optimizers.SGD
}

const (
	v     int = 0
	buf   int = 1
	vPrev int = 2
	vTmp  int = 3
)

func (o *SGD[T]) NewState(shape ...int) any {
	r, c := shape[0], shape[1]

	if o.Mu == 0.0 {
		// Vanilla SGD doesn't require any support structure, this is just to avoid memory allocation
		return []mat.Matrix{mat.NewDense[T](mat.WithShape(r, c))} // v at index 0
	}
	if !o.Nesterov {
		supp := make([]mat.Matrix, 2)
		supp[v] = mat.NewDense[T](mat.WithShape(r, c))
		supp[buf] = mat.NewDense[T](mat.WithShape(r, c))
		return supp
	}
	supp := make([]mat.Matrix, 4)
	supp[v] = mat.NewDense[T](mat.WithShape(r, c))
	supp[buf] = mat.NewDense[T](mat.WithShape(r, c))
	supp[vPrev] = mat.NewDense[T](mat.WithShape(r, c))
	supp[vTmp] = mat.NewDense[T](mat.WithShape(r, c))
	return supp
}

// CalcDelta returns the difference between the current params and where the method wants it to be.
func (o *SGD[T]) CalcDelta(param *nn.Param) mat.Matrix {
	grads := param.Grad()
	supp := param.GetOrSetState(o.NewState).([]mat.Matrix)
	return o.calcDelta(grads, supp)
}

func (o *SGD[T]) calcDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	if o.Mu == 0.0 {
		return o.calcVanillaSGD(grads, supp)
	} else if o.Nesterov {
		return o.calcNesterovMomentumDelta(grads, supp)
	} else {
		return o.calcMomentumDelta(grads, supp)
	}
}

func (o *SGD[T]) calcVanillaSGD(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[v].ProdMatrixScalarInPlace(grads, o.Alpha)
	return supp[v]
}

func (o *SGD[T]) calcMomentumDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[buf].ProdMatrixScalarInPlace(grads, o.Alpha)
	supp[v].ProdScalarInPlace(o.Mu)
	supp[v].AddInPlace(supp[buf])
	return supp[v]
}

func (o *SGD[T]) calcNesterovMomentumDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	supp[buf].ProdMatrixScalarInPlace(grads, o.Alpha)
	supp[vPrev].ProdMatrixScalarInPlace(supp[v], o.Mu)
	supp[v].ProdScalarInPlace(o.Mu)
	supp[v].AddInPlace(supp[buf]) // += grad * alpha
	supp[vTmp].ProdMatrixScalarInPlace(supp[v], 1.0+o.Mu)
	supp[vTmp].SubInPlace(supp[vPrev])
	return supp[vTmp]
}
