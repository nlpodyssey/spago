// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ELU is an operator to perform the ELU activation function.
// ELU(x) = max(0,x) + min(0,α ∗ (exp(x) − 1))
type ELU[O Operand] struct {
	x        O
	alpha    O // scalar
	operands []O
}

// NewELU returns a new ELU Function.
func NewELU[O Operand](x O, alpha O) *ELU[O] {
	return &ELU[O]{
		x:        x,
		alpha:    alpha,
		operands: []O{x, alpha},
	}
}

// Operands returns the list of operands.
func (r *ELU[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *ELU[O]) Forward() mat.Matrix {
	y := r.x.Value().ApplyWithAlpha(elu, r.alpha.Value().Scalar().F64())
	return y
}

// Backward computes the backward pass.
func (r *ELU[O]) Backward(gy mat.Matrix) {
	if !mat.SameDims(r.x.Value(), gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(eluDeriv, r.alpha.Value().Scalar().F64())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
