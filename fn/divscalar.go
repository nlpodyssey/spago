// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// DivScalar is an operator to perform element-wise division with a scalar value.
type DivScalar[O DualValue] struct {
	x1 O
	x2 O // scalar
}

// NewDivScalar returns a new DivScalar Function.
func NewDivScalar[O DualValue](x1 O, x2 O) *DivScalar[O] {
	return &DivScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *DivScalar[O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *DivScalar[O]) Forward() mat.Matrix {
	return r.x1.Value().ProdScalar(1.0 / r.x2.Value().Scalar().F64())
}

// Backward computes the backward pass.
func (r *DivScalar[O]) Backward(gy mat.Matrix) {
	if !mat.SameDims(r.x1.Value(), gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy.ProdScalar(1.0 / r.x2.Value().Scalar().F64()))
	}
	if r.x2.RequiresGrad() {
		x2 := r.x2.Value().Scalar().F64()

		a := r.x1.Value().ProdScalar(1 / -(x2 * x2))
		defer mat.ReleaseMatrix(a)

		b := gy.Prod(a)
		defer mat.ReleaseMatrix(b)

		gx := b.Sum()
		defer mat.ReleaseMatrix(gx)

		r.x2.AccGrad(gx)
	}
}
