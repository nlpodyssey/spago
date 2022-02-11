// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &View[float32]{}

// View is a function to extract a portion of a matrix.
type View[T mat.DType] struct {
	x  Operand[T]
	sx int
	sy int
	lx int // x length
	ly int // y length
}

// NewView returns a new View Function.
func NewView[T mat.DType](x Operand[T], sx, sy, lx, ly int) *View[T] {
	return &View[T]{x: x, sx: sx, sy: sy, lx: lx, ly: ly}
}

// Forward computes the output of the function.
func (r *View[T]) Forward() mat.Matrix[T] {
	y := mat.NewEmptyDense[T](r.lx, r.ly)
	for i := 0; i < r.lx; i++ {
		for j := 0; j < r.ly; j++ {
			y.Set(i, j, r.x.Value().At(i+r.sx, j+r.sy))
		}
	}
	return y
}

// Backward computes the backward pass.
func (r *View[T]) Backward(gy mat.Matrix[T]) {
	if !(gy.Rows() == r.lx && gy.Columns() == r.ly) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for i := 0; i < r.lx; i++ {
			for j := 0; j < r.ly; j++ {
				gx.Set(i+r.sx, j+r.sy, gy.At(i, j))
			}
		}
		r.x.PropagateGrad(gx)
	}
}
