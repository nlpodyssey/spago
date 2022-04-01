// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ReduceMax is an operator to perform reduce-max function.
// It gets the maximum element of the Operand x
type ReduceMax[T mat.DType, O Operand[T]] struct {
	x        O
	argmax   int
	operands []O
}

// NewReduceMax returns a new ReduceMax Function.
func NewReduceMax[T mat.DType, O Operand[T]](x O) *ReduceMax[T, O] {
	return &ReduceMax[T, O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *ReduceMax[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of this function.
func (r *ReduceMax[T, O]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	r.argmax = xv.ArgMax()
	return mat.NewScalar(xv.AtVec(r.argmax))
}

// Backward computes the backward pass.
func (r *ReduceMax[T, O]) Backward(gy mat.Matrix[T]) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewEmptyVecDense[T](r.x.Value().Size())
		defer mat.ReleaseDense(gx)
		for i := range gx.Data() {
			if i == r.argmax {
				gx.Data()[i] = gy.Scalar()
			}
		}
		r.x.PropagateGrad(gx)
	}
}
