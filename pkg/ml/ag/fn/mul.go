// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
)

type Mul struct {
	x1 Operand // matrix
	x2 Operand // vector
}

func NewMul(x1, x2 Operand) *Mul {
	return &Mul{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Mul) Forward() mat.Matrix {
	return r.x1.Value().Mul(r.x2.Value())
}

// TODO: backward of sparse gradients
func (r *Mul) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy.Mul(r.x2.Value().T()))
	}
	if r.x2.RequiresGrad() {
		//r.x2.PropagateGrad(gy.T().Mul(r.x1).T()) // alternative method
		if gy.Columns() == 1 {
			r.x2.PropagateGrad(r.x1.Value().(*mat.Dense).MulT(gy))
		} else {
			r.x2.PropagateGrad(r.x1.Value().T().Mul(gy))
		}
	}
}
