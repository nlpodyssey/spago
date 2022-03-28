// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// ColView is an operator to extract the i-th column from a matrix.
type ColView[T mat.DType, O Operand[T]] struct {
	x        O
	i        int
	operands []O
}

// NewColView extracts the i-th column from the input matrix.
func NewColView[T mat.DType, O Operand[T]](x O, i int) *ColView[T, O] {
	if i < 0 {
		panic("fn: invalid column index")
	}
	return &ColView[T, O]{
		x:        x,
		i:        i,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *ColView[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *ColView[T, O]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	if r.i >= xv.Columns() {
		panic("fn: matrix with not compatible size")
	}
	return xv.ExtractColumn(r.i)
}

// Backward computes the backward pass.
func (r *ColView[T, O]) Backward(gy mat.Matrix[T]) {
	if !(r.x.Value().Rows() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for i := 0; i < r.x.Value().Rows(); i++ {
			gx.Set(i, r.i, gy.AtVec(i))
		}
		r.x.PropagateGrad(gx)
	}
}
