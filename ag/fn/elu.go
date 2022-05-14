// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ELU is an operator to perform the ELU activation function.
// ELU(x) = max(0,x) + min(0,α ∗ (exp(x) − 1))
type ELU[T mat.DType, O Operand[T]] struct {
	x        O
	alpha    O // scalar
	operands []O
}

// NewELU returns a new ELU Function.
func NewELU[T mat.DType, O Operand[T]](x O, alpha O) *ELU[T, O] {
	return &ELU[T, O]{
		x:        x,
		alpha:    alpha,
		operands: []O{x, alpha},
	}
}

// Operands returns the list of operands.
func (r *ELU[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *ELU[T, O]) Forward() mat.Matrix {
	y := r.x.Value().ApplyWithAlpha(elu, r.alpha.Value().Scalar().Float64())
	return y
}

// Backward computes the backward pass.
func (r *ELU[T, O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(eluDeriv, r.alpha.Value().Scalar().Float64())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
