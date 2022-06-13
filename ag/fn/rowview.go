// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// RowView is a function to extract the i-th row from the input matrix.
type RowView[O Operand] struct {
	x O
	i int
}

// NewRowView returns a new RowView Function.
func NewRowView[O Operand](x O, i int) *RowView[O] {
	if i < 0 {
		panic("fn: invalid row index")
	}
	return &RowView[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *RowView[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *RowView[O]) Forward() mat.Matrix {
	return r.x.Value().ExtractRow(r.i)
}

// Backward computes the backward pass.
func (r *RowView[O]) Backward(gy mat.Matrix) {
	if !(r.x.Value().Columns() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for j := 0; j < r.x.Value().Columns(); j++ {
			gx.SetScalar(r.i, j, gy.ScalarAt(0, j))
		}
		r.x.AccGrad(gx)
	}
}
