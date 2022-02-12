// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

var _ Function[float32] = &RowView[float32]{}

// RowView is a function to extract the i-th row from the input matrix.
type RowView[T mat.DType] struct {
	x Operand[T]
	i int
}

// NewRowView returns a new RowView Function.
func NewRowView[T mat.DType](x Operand[T], i int) *RowView[T] {
	if i < 0 {
		panic("fn: invalid row index")
	}
	return &RowView[T]{x: x, i: i}
}

// Forward computes the output of the function.
func (r *RowView[T]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	rows, cols := xv.Dims()
	if r.i >= rows {
		panic("fn: matrix with not compatible size")
	}
	return xv.ExtractRow(r.i).ReshapeInPlace(1, cols)
}

// Backward computes the backward pass.
func (r *RowView[T]) Backward(gy mat.Matrix[T]) {
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
