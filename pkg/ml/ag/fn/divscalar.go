// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &DivScalar{}

// DivScalar is an operator to perform element-wise division with a scalar value.
type DivScalar struct {
	x1 Operand
	x2 Operand // scalar
}

// NewDivScalar returns a new DivScalar Function.
func NewDivScalar(x1, x2 Operand) *DivScalar {
	return &DivScalar{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *DivScalar) Forward() mat.Matrix {
	return r.x1.Value().ProdScalar(1.0 / r.x2.Value().Scalar())
}

// Backward computes the backward pass.
func (r *DivScalar) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy.ProdScalar(1.0 / r.x2.Value().Scalar()))
	}
	if r.x2.RequiresGrad() {
		var gx mat.Float = 0.0
		for i := 0; i < gy.Rows(); i++ {
			for j := 0; j < gy.Columns(); j++ {
				gx += gy.At(i, j) * (r.x1.Value().At(i, j) / (-1.0 * (r.x2.Value().Scalar() * r.x2.Value().Scalar())))
			}
		}
		scalar := mat.NewScalar(gx)
		defer mat.ReleaseDense(scalar)
		r.x2.PropagateGrad(scalar)
	}
}
