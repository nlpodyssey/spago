// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

var _ Function[float32] = &ColView[float32]{}

// ColView is an operator to extract the i-th column from a matrix.
type ColView[T mat.DType] struct {
	x Operand[T]
	i int
}

// NewColView extracts the i-th column from the input matrix.
func NewColView[T mat.DType](x Operand[T], i int) *ColView[T] {
	if i < 0 {
		panic("fn: invalid column index")
	}
	return &ColView[T]{x: x, i: i}
}

// Forward computes the output of the function.
func (r *ColView[T]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	rows, cols := xv.Dims()
	if r.i >= cols {
		panic("fn: matrix with not compatible size")
	}
	return xv.ExtractColumn(r.i).ReshapeInPlace(1, rows)
}

// Backward computes the backward pass.
func (r *ColView[T]) Backward(gy mat.Matrix[T]) {
	if !(r.x.Value().Rows() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for i := 0; i < r.x.Value().Rows(); i++ {
			gx.Set(i, r.i, gy.At(0, i))
		}
		r.x.PropagateGrad(gx)
	}
}
