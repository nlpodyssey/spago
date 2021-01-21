// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &Pow{}

// Pow is an operator to perform element-wise pow function.
type Pow struct {
	x     Operand
	power mat.Float
}

// NewPow returns a new Pow Function.
func NewPow(x Operand, power mat.Float) *Pow {
	return &Pow{x: x, power: power}
}

// Forward computes the output of the function.
func (r *Pow) Forward() mat.Matrix {
	return r.x.Value().Pow(r.power)
}

// Backward computes the backward pass.
func (r *Pow) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().Pow(r.power - 1)
		defer mat.ReleaseMatrix(gx)
		gx.ProdScalarInPlace(r.power).ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
