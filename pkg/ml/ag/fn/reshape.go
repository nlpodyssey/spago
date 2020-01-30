// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"saientist.dev/spago/pkg/mat"
)

type Reshape struct {
	x    Operand
	rows int
	cols int
}

func NewReshape(x Operand, r, c int) *Reshape {
	return &Reshape{x: x, rows: r, cols: c}
}

// Forward computes the output of the node.
func (r *Reshape) Forward() mat.Matrix {
	return r.x.Value().Reshape(r.rows, r.cols)
}

func (r *Reshape) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		r.x.PropagateGrad(gy.Reshape(r.x.Value().Dims()))
	}
}
