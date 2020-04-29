// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &Transpose{}

type Transpose struct {
	x Operand
}

func NewTranspose(x Operand) *Transpose {
	return &Transpose{x: x}
}

// Forward computes the output of the node.
func (r *Transpose) Forward() mat.Matrix {
	return r.x.Value().T()
}

func (r *Transpose) Backward(gy mat.Matrix) {
	if r.x.Value().Columns() != gy.Rows() && r.x.Value().Rows() != gy.Columns() {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.T()
		defer mat.ReleaseDense(gx.(*mat.Dense))
		r.x.PropagateGrad(gx)
	}
}
