// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// SELU function: f(x) = scale ∗ (max(0,x) + min(0, α ∗ (exp(x) − 1)))
type SELU[O mat.Tensor] struct {
	x     O
	alpha O // scalar
	scale O // scalar
}

// NewSELU returns a new SELU Function.
func NewSELU[O mat.Tensor](x O, alpha, scale O) *SELU[O] {
	return &SELU[O]{
		x:     x,
		alpha: alpha,
		scale: scale,
	}
}

// Operands returns the list of operands.
func (r *SELU[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x, r.alpha, r.scale}
}

// Forward computes the output of the function.
func (r *SELU[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).ApplyWithAlpha(
		selu,
		r.alpha.Value().Item().F64(),
		r.scale.Value().Item().F64(),
	), nil
}

// Backward computes the backward pass.
func (r *SELU[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ApplyWithAlpha(
			seluDeriv,
			r.alpha.Value().Item().F64(),
			r.scale.Value().Item().F64(),
		)
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
