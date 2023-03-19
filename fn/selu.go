// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// SELU function: f(x) = scale ∗ (max(0,x) + min(0, α ∗ (exp(x) − 1)))
type SELU[O DualValue] struct {
	x     O
	alpha O // scalar
	scale O // scalar
}

// NewSELU returns a new SELU Function.
func NewSELU[O DualValue](x O, alpha, scale O) *SELU[O] {
	return &SELU[O]{
		x:     x,
		alpha: alpha,
		scale: scale,
	}
}

// Operands returns the list of operands.
func (r *SELU[O]) Operands() []O {
	return []O{r.x, r.alpha, r.scale}
}

// Forward computes the output of the function.
func (r *SELU[O]) Forward() mat.Matrix {
	return r.x.Value().ApplyWithAlpha(
		selu,
		r.alpha.Value().Scalar().F64(),
		r.scale.Value().Scalar().F64(),
	)
}

// Backward computes the backward pass.
func (r *SELU[O]) Backward(gy mat.Matrix) {
	if !mat.SameDims(r.x.Value(), gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(
			seluDeriv,
			r.alpha.Value().Scalar().F64(),
			r.scale.Value().Scalar().F64(),
		)
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
