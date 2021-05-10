// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &Prod{}

// Prod is an operator to perform element-wise product over two values.
type Prod struct {
	x1 Operand
	x2 Operand
}

// NewProd returns a new Prod Function.
func NewProd(x1, x2 Operand) *Prod {
	return &Prod{x1: x1, x2: x2}
}

// Square is an operator to perform element-wise square.
type Square struct {
	*Prod
}

// NewSquare returns a new Prod Function with both operands set to the given value x.
func NewSquare(x Operand) *Square {
	return &Square{Prod: &Prod{x1: x, x2: x}}
}

// Forward computes the output of the node.
func (r *Prod) Forward() mat.Matrix {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	return x1v.Prod(x2v)
}

// Backward computes the backward pass.
func (r *Prod) Backward(gy mat.Matrix) {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, gy) || mat.VectorsOfSameSize(x1v, gy)) &&
		!(mat.SameDims(x2v, gy) || mat.VectorsOfSameSize(x2v, gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		gx := r.x2.Value().Prod(gy)
		defer mat.ReleaseMatrix(gx)
		r.x1.PropagateGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gx := r.x1.Value().Prod(gy)
		defer mat.ReleaseMatrix(gx)
		r.x2.PropagateGrad(gx)
	}
}
