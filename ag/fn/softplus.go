// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// SoftPlus function: f(x) = 1 / β ∗ log(1 + exp(β ∗ x))
type SoftPlus[T mat.DType, O Operand[T]] struct {
	x         O
	beta      O
	threshold O
	operands  []O
}

// NewSoftPlus returns a new SoftPlus Function.
func NewSoftPlus[T mat.DType, O Operand[T]](x O, beta, threshold O) *SoftPlus[T, O] {
	return &SoftPlus[T, O]{
		x:         x,
		beta:      beta,
		threshold: threshold,
		operands:  []O{x, beta, threshold},
	}
}

// Operands returns the list of operands.
func (r *SoftPlus[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *SoftPlus[T, O]) Forward() mat.Matrix[T] {
	y := r.x.Value().ApplyWithAlpha(softPlus[T], r.beta.Value().Scalar(), r.threshold.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SoftPlus[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(softPlusDeriv[T], r.beta.Value().Scalar(), r.threshold.Value().Scalar())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
