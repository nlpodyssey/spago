// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// CELU is an operator to perform the CELU activation.
// CELU(x) = max(0,x) + min(0,α ∗ (exp(x/α) − 1))
type CELU[O Operand] struct {
	x        O
	alpha    O // scalar
	operands []O
}

// NewCELU returns a new CELU Function.
func NewCELU[O Operand](x O, alpha O) *CELU[O] {
	return &CELU[O]{
		x:        x,
		alpha:    alpha,
		operands: []O{x, alpha},
	}
}

// Operands returns the list of operands.
func (r *CELU[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *CELU[O]) Forward() mat.Matrix {
	return r.x.Value().ApplyWithAlpha(celu, r.alpha.Value().Scalar().F64())
}

// Backward computes the backward pass.
func (r *CELU[O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(celuDeriv, r.alpha.Value().Scalar().F64())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
