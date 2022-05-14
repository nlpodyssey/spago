// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// ColView is an operator to extract the i-th column from a matrix.
type ColView[O Operand] struct {
	x        O
	i        int
	operands []O
}

// NewColView extracts the i-th column from the input matrix.
func NewColView[O Operand](x O, i int) *ColView[O] {
	if i < 0 {
		panic("fn: invalid column index")
	}
	return &ColView[O]{
		x:        x,
		i:        i,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *ColView[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *ColView[O]) Forward() mat.Matrix {
	return r.x.Value().ExtractColumn(r.i)
}

// Backward computes the backward pass.
func (r *ColView[O]) Backward(gy mat.Matrix) {
	if !(r.x.Value().Rows() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for i := 0; i < r.x.Value().Rows(); i++ {
			gx.SetScalar(i, r.i, gy.ScalarAtVec(i))
		}
		r.x.AccGrad(gx)
	}
}
