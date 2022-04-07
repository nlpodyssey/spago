// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// SELU function: f(x) = scale ∗ (max(0,x) + min(0, α ∗ (exp(x) − 1)))
type SELU[T mat.DType, O Operand[T]] struct {
	x        O
	alpha    O // scalar
	scale    O // scalar
	operands []O
}

// NewSELU returns a new SELU Function.
func NewSELU[T mat.DType, O Operand[T]](x O, alpha, scale O) *SELU[T, O] {
	return &SELU[T, O]{
		x:        x,
		alpha:    alpha,
		scale:    scale,
		operands: []O{x, alpha, scale},
	}
}

// Operands returns the list of operands.
func (r *SELU[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *SELU[T, O]) Forward() mat.Matrix[T] {
	y := r.x.Value().ApplyWithAlpha(selu[T], r.alpha.Value().Scalar(), r.scale.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SELU[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(seluDeriv[T], r.alpha.Value().Scalar(), r.scale.Value().Scalar())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
