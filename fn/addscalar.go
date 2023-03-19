// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// AddScalar is an operator to perform element-wise addition over two values.
type AddScalar[O DualValue] struct {
	x1 O
	x2 O // scalar
}

// NewAddScalar returns a new AddScalar Function.
func NewAddScalar[O DualValue](x1, x2 O) *AddScalar[O] {
	return &AddScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *AddScalar[O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the function.
// It doesn't backward on the scalar value x2.
func (r *AddScalar[O]) Forward() mat.Matrix {
	return r.x1.Value().AddScalar(r.x2.Value().Scalar().F64())
}

// Backward computes the backward pass.
func (r *AddScalar[O]) Backward(gy mat.Matrix) {
	if !mat.SameDims(r.x1.Value(), gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy)
	}
	if r.x2.RequiresGrad() {
		gx := gy.Sum()
		defer mat.ReleaseMatrix(gx)
		r.x2.AccGrad(gx)
	}
}
