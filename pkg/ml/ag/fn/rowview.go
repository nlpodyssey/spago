// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import mat "github.com/nlpodyssey/spago/pkg/mat32"

var _ Function = &RowView{}

// RowView is a function to extract the i-th row from the input matrix.
type RowView struct {
	x Operand
	i int
}

// NewRowView returns a new RowView Function.
func NewRowView(x Operand, i int) *RowView {
	if i < 0 {
		panic("fn: invalid row index")
	}
	return &RowView{x: x, i: i}
}

// Forward computes the output of the function.
func (r *RowView) Forward() mat.Matrix {
	xv := r.x.Value()
	rows, cols := xv.Dims()
	if r.i >= rows {
		panic("fn: matrix with not compatible size")
	}
	y := mat.GetDenseWorkspace(1, cols)
	for j := 0; j < cols; j++ {
		y.Set(0, j, xv.At(r.i, j))
	}
	return y
}

// Backward computes the backward pass.
func (r *RowView) Backward(gy mat.Matrix) {
	if !(r.x.Value().Columns() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewEmptyDense(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		for j := 0; j < r.x.Value().Columns(); j++ {
			gx.Set(r.i, j, gy.At(0, j))
		}
		r.x.PropagateGrad(gx)
	}
}
