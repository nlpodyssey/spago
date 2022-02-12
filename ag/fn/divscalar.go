// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &DivScalar[float32]{}

// DivScalar is an operator to perform element-wise division with a scalar value.
type DivScalar[T mat.DType] struct {
	x1 Operand[T]
	x2 Operand[T] // scalar
}

// NewDivScalar returns a new DivScalar Function.
func NewDivScalar[T mat.DType](x1, x2 Operand[T]) *DivScalar[T] {
	return &DivScalar[T]{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *DivScalar[T]) Forward() mat.Matrix[T] {
	return r.x1.Value().ProdScalar(1.0 / r.x2.Value().Scalar())
}

// Backward computes the backward pass.
func (r *DivScalar[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy.ProdScalar(1.0 / r.x2.Value().Scalar()))
	}
	if r.x2.RequiresGrad() {
		var gx T = 0.0
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
