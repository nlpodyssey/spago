// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &Threshold[float32]{}

// Threshold function: f(x) = x if x > threshold; k otherwise.
type Threshold[T mat.DType] struct {
	x         Operand[T]
	threshold Operand[T] // scalar
	k         Operand[T] // scalar
}

// NewThreshold returns a new Threshold Function.
func NewThreshold[T mat.DType](x, threshold, k Operand[T]) *Threshold[T] {
	return &Threshold[T]{x: x, threshold: threshold, k: k}
}

// Forward computes the output of the function.
func (r *Threshold[T]) Forward() mat.Matrix[T] {
	y := mat.GetDensePool[T]().Get(r.x.Value().Dims())
	y.ApplyWithAlphaInPlace(threshold[T], r.x.Value(), r.threshold.Value().Scalar(), r.k.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *Threshold[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[T]().Get(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlphaInPlace(thresholdDeriv[T], r.x.Value(), r.threshold.Value().Scalar(), r.k.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
