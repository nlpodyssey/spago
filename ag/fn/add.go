// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Add is an operator to perform element-wise sum over two values.
// y = x1 + x2
type Add[T mat.DType, O Operand[T]] struct {
	x1       O
	x2       O
	operands []O
}

// NewAdd returns a new Add Function.
func NewAdd[T mat.DType, O Operand[T]](x1, x2 O) *Add[T, O] {
	return &Add[T, O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Operands returns the list of operands.
func (r *Add[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Add[T, O]) Forward() mat.Matrix[T] {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if x1v == nil {
		x1v = x2v.ZerosLike()
		defer mat.ReleaseMatrix(x1v)
	}
	return x1v.Add(x2v)
}

// Backward computes the backward pass.
func (r *Add[T, O]) Backward(gy mat.Matrix[T]) {
	if r.x1.RequiresGrad() {
		x1v := r.x1.Value()
		if !(mat.SameDims(x1v, gy) || mat.VectorsOfSameSize(x1v, gy)) {
			panic("fn: matrices with not compatible size")
		}
		r.x1.PropagateGrad(gy)
	}
	if r.x2.RequiresGrad() {
		x2v := r.x2.Value()
		if !(mat.SameDims(x2v, gy) || mat.VectorsOfSameSize(x2v, gy)) {
			panic("fn: matrices with not compatible size")
		}
		r.x2.PropagateGrad(gy)
	}
}
