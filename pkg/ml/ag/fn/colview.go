// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

type ColView struct {
	x Operand
	i int
}

// Extract the i-th column from the input matrix
func NewColView(x Operand, i int) *ColView {
	return &ColView{x: x, i: i}
}

// Forward computes the output of the function.
func (r *ColView) Forward() mat.Matrix {
	y := mat.NewEmptyDense(1, r.x.Value().Rows())
	for i := 0; i < r.x.Value().Rows(); i++ {
		y.Set(r.x.Value().At(i, r.i), 0, i)
	}
	return y
}

func (r *ColView) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := mat.NewEmptyDense(r.x.Value().Dims())
		for i := 0; i < r.x.Value().Rows(); i++ {
			gx.Set(gy.At(0, i), i, r.i)
		}
		r.x.PropagateGrad(gx)
	}
}
