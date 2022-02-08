// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &SoftShrink{}

// SoftShrink function: f(x) = x − λ if x > λ; x + λ if x < −λ; 0 otherwise.
type SoftShrink struct {
	x      Operand
	lambda Operand // scalar
}

// NewSoftShrink returns a new SoftShrink Function.
func NewSoftShrink(x, lambda Operand) *SoftShrink {
	return &SoftShrink{x: x, lambda: lambda}
}

// Forward computes the output of the function.
func (r *SoftShrink) Forward() mat.Matrix[mat.Float] {
	y := mat.GetDensePool[mat.Float]().Get(r.x.Value().Dims())
	y.ApplyWithAlpha(softShrink, r.x.Value(), r.lambda.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SoftShrink) Backward(gy mat.Matrix[mat.Float]) {
	if !(r.x.Value().SameDims(gy) || r.x.Value().VectorOfSameSize(gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[mat.Float]().Get(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(softShrinkDeriv, r.x.Value(), r.lambda.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
