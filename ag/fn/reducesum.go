// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ReduceSum is an operator to perform reduce-sum function.
type ReduceSum[O Operand] struct {
	x        O
	operands []O
}

// NewReduceSum returns a new ReduceSum Function.
func NewReduceSum[O Operand](x O) *ReduceSum[O] {
	return &ReduceSum[O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *ReduceSum[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of this function.
func (r *ReduceSum[O]) Forward() mat.Matrix {
	return r.x.Value().Sum()
}

// Backward computes the backward pass.
func (r *ReduceSum[O]) Backward(gy mat.Matrix) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		x := r.x.Value()
		gx := x.NewInitVec(x.Size(), gy.Scalar())
		defer mat.ReleaseMatrix(gx)
		r.x.AccGrad(gx)
	}
}
