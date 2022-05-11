// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ReduceSum is an operator to perform reduce-sum function.
type ReduceSum[T mat.DType, O Operand[T]] struct {
	x        O
	operands []O
}

// NewReduceSum returns a new ReduceSum Function.
func NewReduceSum[T mat.DType, O Operand[T]](x O) *ReduceSum[T, O] {
	return &ReduceSum[T, O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *ReduceSum[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of this function.
func (r *ReduceSum[T, O]) Forward() mat.Matrix[T] {
	return r.x.Value().Sum()
}

// Backward computes the backward pass.
func (r *ReduceSum[T, O]) Backward(gy mat.Matrix[T]) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar())
		defer mat.ReleaseDense(gx)
		r.x.AccGrad(gx)
	}
}
