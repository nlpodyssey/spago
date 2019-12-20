// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
)

// Element-wise division over two values.
type Div struct {
	x1 Operand
	x2 Operand
}

func NewDiv(x1, x2 Operand) *Div {
	return &Div{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Div) Forward() mat.Matrix {
	return r.x1.Value().Div(r.x2.Value())
}

func (r *Div) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy.Div(r.x2.Value()))
	}
	if r.x2.RequiresGrad() {
		r.x2.PropagateGrad(r.x1.Value().Prod(gy).ProdScalar(-1).Div(r.x2.Value().Prod(r.x2.Value())))
	}
}
