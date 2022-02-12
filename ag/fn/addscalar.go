// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &AddScalar[float32]{}

// AddScalar is an operator to perform element-wise addition over two values.
type AddScalar[T mat.DType] struct {
	x1 Operand[T]
	x2 Operand[T] // scalar
}

// NewAddScalar returns a new AddScalar Function.
func NewAddScalar[T mat.DType](x1, x2 Operand[T]) *AddScalar[T] {
	return &AddScalar[T]{x1: x1, x2: x2}
}

// Forward computes the output of the function.
// It doesn't backward on the scalar value x2.
func (r *AddScalar[T]) Forward() mat.Matrix[T] {
	return r.x1.Value().AddScalar(r.x2.Value().Scalar())
}

// Backward computes the backward pass.
func (r *AddScalar[T]) Backward(gy mat.Matrix[T]) {
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
