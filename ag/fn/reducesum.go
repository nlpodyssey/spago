// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &ReduceSum[float32]{}

// ReduceSum is an operator to perform reduce-sum function.
type ReduceSum[T mat.DType] struct {
	x Operand[T]
}

// NewReduceSum returns a new ReduceSum Function.
func NewReduceSum[T mat.DType](x Operand[T]) *ReduceSum[T] {
	return &ReduceSum[T]{x: x}
}

// Forward computes the output of this function.
func (r *ReduceSum[T]) Forward() mat.Matrix[T] {
	return mat.NewScalar(r.x.Value().Sum())
}

// Backward computes the backward pass.
func (r *ReduceSum[T]) Backward(gy mat.Matrix[T]) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar())
		defer mat.ReleaseDense(gx)
		r.x.PropagateGrad(gx)
	}
}
