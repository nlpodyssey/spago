// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &SELU{}

// SELU function: f(x) = scale ∗ (max(0,x) + min(0, α ∗ (exp(x) − 1)))
type SELU struct {
	x     Operand
	alpha Operand // scalar
	scale Operand // scalar
}

// NewSELU returns a new SELU Function.
func NewSELU(x, alpha, scale Operand) *SELU {
	return &SELU{x: x, alpha: alpha, scale: scale}
}

// Forward computes the output of the function.
func (r *SELU) Forward() mat.Matrix[mat.Float] {
	y := mat.GetDensePool[mat.Float]().Get(r.x.Value().Dims())
	y.ApplyWithAlpha(selu, r.x.Value(), r.alpha.Value().Scalar(), r.scale.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SELU) Backward(gy mat.Matrix[mat.Float]) {
	if !(r.x.Value().SameDims(gy) || r.x.Value().VectorOfSameSize(gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[mat.Float]().Get(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(seluDeriv, r.x.Value(), r.alpha.Value().Scalar(), r.scale.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
