// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

type RowView struct {
	x Operand
	i int
}

// Extract the i-th row from the input matrix
func NewRowView(x Operand, i int) *RowView {
	return &RowView{x: x, i: i}
}

// Forward computes the output of the function.
func (r *RowView) Forward() mat.Matrix {
	y := mat.NewEmptyDense(1, r.x.Value().Columns())
	for j := 0; j < r.x.Value().Columns(); j++ {
		y.Set(r.x.Value().At(r.i, j), 0, j)
	}
	return y
}

func (r *RowView) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := mat.NewEmptyDense(r.x.Value().Dims())
		for j := 0; j < r.x.Value().Columns(); j++ {
			gx.Set(gy.At(0, j), r.i, j)
		}
		r.x.PropagateGrad(gx)
	}
}
