// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &Reshape[float32]{}

// Reshape is a Function which reshapes an operand into a new matrix of given
// rows × columns size.
type Reshape[T mat.DType] struct {
	x    Operand[T]
	rows int
	cols int
}

// NewReshape returns a new Reshape Function.
func NewReshape[T mat.DType](x Operand[T], r, c int) *Reshape[T] {
	return &Reshape[T]{x: x, rows: r, cols: c}
}

// Forward computes the output of the node.
func (r *Reshape[T]) Forward() mat.Matrix[T] {
	if r.x.Value().Size() != r.rows*r.cols {
		panic("fn: incompatible sizes")
	}
	return r.x.Value().Reshape(r.rows, r.cols)
}

// Backward computes the backward pass.
func (r *Reshape[T]) Backward(gy mat.Matrix[T]) {
	if gy.Columns() != r.cols && gy.Rows() != r.rows {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
