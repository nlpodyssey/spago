// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// AddScalar is an operator to perform element-wise addition over two values.
type AddScalar[T mat.DType, O Operand[T]] struct {
	x1 O
	x2 O // scalar
}

// NewAddScalar returns a new AddScalar Function.
func NewAddScalar[T mat.DType, O Operand[T]](x1, x2 O) *AddScalar[T, O] {
	return &AddScalar[T, O]{x1: x1, x2: x2}
}

// Operands returns the list of operands.
func (r *AddScalar[T, O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the function.
// It doesn't backward on the scalar value x2.
func (r *AddScalar[T, O]) Forward() mat.Matrix[T] {
	return r.x1.Value().AddScalar(r.x2.Value().Scalar())
}

// Backward computes the backward pass.
func (r *AddScalar[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy)
	}
	if r.x2.RequiresGrad() {
		gx := mat.NewScalar(gy.Sum())
		defer mat.ReleaseDense(gx)
		r.x2.PropagateGrad(gx)
	}
}
