// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// DivScalar is an operator to perform element-wise division with a scalar value.
type DivScalar[O Operand] struct {
	x1       O
	x2       O // scalar
	operands []O
}

// NewDivScalar returns a new DivScalar Function.
func NewDivScalar[O Operand](x1 O, x2 O) *DivScalar[O] {
	return &DivScalar[O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Operands returns the list of operands.
func (r *DivScalar[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *DivScalar[O]) Forward() mat.Matrix {
	return r.x1.Value().ProdScalar(1.0 / r.x2.Value().Scalar().Float64())
}

// Backward computes the backward pass.
func (r *DivScalar[O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy.ProdScalar(1.0 / r.x2.Value().Scalar().Float64()))
	}
	if r.x2.RequiresGrad() {
		x2 := r.x2.Value().Scalar().Float64()

		a := r.x1.Value().ProdScalar(1 / -(x2 * x2))
		defer mat.ReleaseMatrix(a)

		b := gy.Prod(a)
		defer mat.ReleaseMatrix(b)

		gx := b.Sum()
		defer mat.ReleaseMatrix(gx)

		r.x2.AccGrad(gx)
	}
}
