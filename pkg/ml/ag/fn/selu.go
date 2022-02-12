// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &SELU[float32]{}

// SELU function: f(x) = scale ∗ (max(0,x) + min(0, α ∗ (exp(x) − 1)))
type SELU[T mat.DType] struct {
	x     Operand[T]
	alpha Operand[T] // scalar
	scale Operand[T] // scalar
}

// NewSELU returns a new SELU Function.
func NewSELU[T mat.DType](x, alpha, scale Operand[T]) *SELU[T] {
	return &SELU[T]{x: x, alpha: alpha, scale: scale}
}

// Forward computes the output of the function.
func (r *SELU[T]) Forward() mat.Matrix[T] {
	y := mat.GetDensePool[T]().Get(r.x.Value().Dims())
	y.ApplyWithAlphaInPlace(selu[T], r.x.Value(), r.alpha.Value().Scalar(), r.scale.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SELU[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[T]().Get(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlphaInPlace(seluDeriv[T], r.x.Value(), r.alpha.Value().Scalar(), r.scale.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
