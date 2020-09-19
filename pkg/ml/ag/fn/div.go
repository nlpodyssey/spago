// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &Div{}

// Div is an operator to perform element-wise division over two values.
type Div struct {
	x1 Operand
	x2 Operand
}

func NewDiv(x1, x2 Operand) *Div {
	return &Div{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Div) Forward() mat.Matrix {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	return x1v.Div(x2v)
}

func (r *Div) Backward(gy mat.Matrix) {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, gy) || mat.VectorsOfSameSize(x1v, gy)) &&
		!(mat.SameDims(x2v, gy) || mat.VectorsOfSameSize(x2v, gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		gx := gy.Div(r.x2.Value())
		defer mat.ReleaseDense(gx.(*mat.Dense))
		r.x1.PropagateGrad(gx)
	}
	if r.x2.RequiresGrad() {
		x2sq := r.x2.Value().Prod(r.x2.Value())
		defer mat.ReleaseDense(x2sq.(*mat.Dense))
		gx := r.x1.Value().Prod(gy)
		defer mat.ReleaseDense(gx.(*mat.Dense))
		gx.ProdScalarInPlace(-1)
		gx.DivInPlace(x2sq)
		r.x2.PropagateGrad(gx)
	}
}
