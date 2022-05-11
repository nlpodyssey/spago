// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ReduceMean is an operator to perform reduce-mean function.
type ReduceMean[T mat.DType, O Operand[T]] struct {
	x        O
	operands []O
}

// NewReduceMean returns a new ReduceMean Function.
func NewReduceMean[T mat.DType, O Operand[T]](x O) *ReduceMean[T, O] {
	return &ReduceMean[T, O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *ReduceMean[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of this node.
func (r *ReduceMean[T, O]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	return xv.Sum().ProdScalarInPlace(1 / float64(xv.Size()))
}

// Backward computes the backward pass.
func (r *ReduceMean[T, O]) Backward(gy mat.Matrix[T]) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar()/T(r.x.Value().Size()))
		defer mat.ReleaseDense(gx)
		r.x.AccGrad(gx)
	}
}
