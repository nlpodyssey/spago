// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// SoftShrink function: f(x) = x − λ if x > λ; x + λ if x < −λ; 0 otherwise.
type SoftShrink[T mat.DType, O Operand[T]] struct {
	x      O
	lambda O // scalar
}

// NewSoftShrink returns a new SoftShrink Function.
func NewSoftShrink[T mat.DType, O Operand[T]](x O, lambda O) *SoftShrink[T, O] {
	return &SoftShrink[T, O]{x: x, lambda: lambda}
}

// Operands returns the list of operands.
func (r *SoftShrink[T, O]) Operands() []O {
	return []O{r.x, r.lambda}
}

// Forward computes the output of the function.
func (r *SoftShrink[T, O]) Forward() mat.Matrix[T] {
	y := r.x.Value().ApplyWithAlpha(softShrink[T], r.lambda.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SoftShrink[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(softShrinkDeriv[T], r.lambda.Value().Scalar())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
