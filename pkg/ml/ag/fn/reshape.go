// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &Reshape{}

// Reshape is a Function which reshapes an operand into a new matrix of given
// rows × columns size.
type Reshape struct {
	x    Operand
	rows int
	cols int
}

// NewReshape returns a new Reshape Function.
func NewReshape(x Operand, r, c int) *Reshape {
	return &Reshape{x: x, rows: r, cols: c}
}

// Forward computes the output of the node.
func (r *Reshape) Forward() mat.Matrix {
	if r.x.Value().Size() != r.rows*r.cols {
		panic("fn: incompatible sizes")
	}
	return r.x.Value().Reshape(r.rows, r.cols)
}

// Backward computes the backward pass.
func (r *Reshape) Backward(gy mat.Matrix) {
	if gy.Columns() != r.cols && gy.Rows() != r.rows {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
