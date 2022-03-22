// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// RowView is a function to extract the i-th row from the input matrix.
type RowView[T mat.DType, O Operand[T]] struct {
	x O
	i int
}

// NewRowView returns a new RowView Function.
func NewRowView[T mat.DType, O Operand[T]](x O, i int) *RowView[T, O] {
	if i < 0 {
		panic("fn: invalid row index")
	}
	return &RowView[T, O]{x: x, i: i}
}

// Operands returns the list of operands.
func (r *RowView[T, O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *RowView[T, O]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	if r.i >= xv.Rows() {
		panic("fn: matrix with not compatible size")
	}
	return xv.ExtractRow(r.i)
}

// Backward computes the backward pass.
func (r *RowView[T, O]) Backward(gy mat.Matrix[T]) {
	if !(r.x.Value().Columns() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for j := 0; j < r.x.Value().Columns(); j++ {
			gx.Set(r.i, j, gy.At(0, j))
		}
		r.x.PropagateGrad(gx)
	}
}
