// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Reshape is a Function which reshapes an operand into a new matrix of given
// rows × columns size.
type Reshape[T mat.DType, O Operand[T]] struct {
	x        O
	rows     int
	cols     int
	operands []O
}

// NewReshape returns a new Reshape Function.
func NewReshape[T mat.DType, O Operand[T]](x O, r, c int) *Reshape[T, O] {
	return &Reshape[T, O]{
		x:        x,
		rows:     r,
		cols:     c,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *Reshape[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the node.
func (r *Reshape[T, O]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	if xv.Size() != r.rows*r.cols {
		panic("fn: incompatible sizes")
	}
	return xv.Reshape(r.rows, r.cols)
}

// Backward computes the backward pass.
func (r *Reshape[T, O]) Backward(gy mat.Matrix[T]) {
	if gy.Columns() != r.cols && gy.Rows() != r.rows {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
