// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// SoftPlus function: f(x) = 1 / β ∗ log(1 + exp(β ∗ x))
type SoftPlus[O mat.Tensor] struct {
	x         O
	beta      O
	threshold O
}

// NewSoftPlus returns a new SoftPlus Function.
func NewSoftPlus[O mat.Tensor](x O, beta, threshold O) *SoftPlus[O] {
	return &SoftPlus[O]{
		x:         x,
		beta:      beta,
		threshold: threshold,
	}
}

// Operands returns the list of operands.
func (r *SoftPlus[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x, r.beta, r.threshold}
}

// Forward computes the output of the function.
func (r *SoftPlus[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).ApplyWithAlpha(
		softPlus,
		r.beta.Value().Item().F64(),
		r.threshold.Value().Item().F64(),
	), nil
}

// Backward computes the backward pass.
func (r *SoftPlus[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ApplyWithAlpha(
			softPlusDeriv,
			r.beta.Value().Item().F64(),
			r.threshold.Value().Item().F64(),
		)
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
