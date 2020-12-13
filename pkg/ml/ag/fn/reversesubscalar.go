// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

var _ Function = &ReverseSubScalar{}

// ReverseSubScalar is the element-wise subtraction function over two values.
type ReverseSubScalar struct {
	x1 Operand
	x2 Operand // scalar
}

func NewReverseSubScalar(x1, x2 Operand) *ReverseSubScalar {
	return &ReverseSubScalar{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *ReverseSubScalar) Forward() mat.Matrix {
	return mat.NewInitDense(r.x1.Value().Rows(), r.x1.Value().Columns(), r.x2.Value().Scalar()).Sub(r.x1.Value())
}

func (r *ReverseSubScalar) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		gx := gy.ProdScalar(-1.0)
		defer mat.ReleaseDense(gx.(*mat.Dense))
		r.x1.PropagateGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gx := 0.0
		for i := 0; i < gy.Rows(); i++ {
			for j := 0; j < gy.Columns(); j++ {
				gx += gy.At(i, j)
			}
		}
		scalar := mat.NewScalar(gx)
		defer mat.ReleaseDense(scalar)
		r.x2.PropagateGrad(scalar)
	}
}
