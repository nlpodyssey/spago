// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgd

import (
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/optimizers/gd"
)

type Config struct {
	LR       float64
	Mu       float64
	Nesterov bool
}

func NewConfig(lr, momentum float64, nesterov bool) Config {
	return Config{
		LR:       lr,
		Mu:       momentum,
		Nesterov: nesterov,
	}
}

type SGD struct {
	Config
	Alpha float64
}

func New(c Config) *SGD {
	return &SGD{Config: c, Alpha: c.LR}
}

func (o *SGD) Name() gd.MethodName {
	return gd.SGD
}

const (
	v     int = 0
	buf   int = 1
	vPrev int = 2
	vTmp  int = 3
)

func (o *SGD) NewSupport(r, c int) *gd.Support {
	if o.Mu == 0.0 {
		// Vanilla SGD doesn't require any support structure, this is just to avoid memory allocation
		return &gd.Support{Name: o.Name(), Data: []mat.Matrix{mat.NewEmptyDense(r, c)}} // v at index 0
	} else {
		if !o.Nesterov {
			supp := make([]mat.Matrix, 2, 2)
			supp[v] = mat.NewEmptyDense(r, c)
			supp[buf] = mat.NewEmptyDense(r, c)
			return &gd.Support{Name: o.Name(), Data: supp}
		} else {
			supp := make([]mat.Matrix, 4, 4)
			supp[v] = mat.NewEmptyDense(r, c)
			supp[buf] = mat.NewEmptyDense(r, c)
			supp[vPrev] = mat.NewEmptyDense(r, c)
			supp[vTmp] = mat.NewEmptyDense(r, c)
			return &gd.Support{Name: o.Name(), Data: supp}
		}
	}
}

func (o *SGD) Delta(param gd.Optimizable) mat.Matrix {
	return o.calcDelta(param.Grad(), param.GetOrSetSupport(o).Data)
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
