// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Reshape is a Function which reshapes an operand into a new matrix of given
// rows × columns size.
type Reshape[O Operand] struct {
	x    O
	rows int
	cols int
}

// NewReshape returns a new Reshape Function.
func NewReshape[O Operand](x O, r, c int) *Reshape[O] {
	return &Reshape[O]{
		x:    x,
		rows: r,
		cols: c,
	}
}

// Operands returns the list of operands.
func (r *Reshape[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the node.
func (r *Reshape[O]) Forward() mat.Matrix {
	return r.x.Value().Reshape(r.rows, r.cols)
}

// Backward computes the backward pass.
func (r *Reshape[O]) Backward(gy mat.Matrix) {
	if gy.Columns() != r.cols && gy.Rows() != r.rows {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.AccGrad(gx)
	}
}
