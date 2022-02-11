// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &SoftPlus[float32]{}

// SoftPlus function: f(x) = 1 / β ∗ log(1 + exp(β ∗ x))
type SoftPlus[T mat.DType] struct {
	x         Operand[T]
	beta      Operand[T]
	threshold Operand[T]
}

// NewSoftPlus returns a new SoftPlus Function.
func NewSoftPlus[T mat.DType](x, beta, threshold Operand[T]) *SoftPlus[T] {
	return &SoftPlus[T]{x: x, beta: beta, threshold: threshold}
}

// Forward computes the output of the function.
func (r *SoftPlus[T]) Forward() mat.Matrix[T] {
	y := mat.GetDensePool[T]().Get(r.x.Value().Dims())
	y.ApplyWithAlpha(softPlus[T], r.x.Value(), r.beta.Value().Scalar(), r.threshold.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SoftPlus[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[T]().Get(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(softPlusDeriv[T], r.x.Value(), r.beta.Value().Scalar(), r.threshold.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
