// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// SoftPlus function: f(x) = 1 / β ∗ log(1 + exp(β ∗ x))
type SoftPlus[O DualValue] struct {
	x         O
	beta      O
	threshold O
}

// NewSoftPlus returns a new SoftPlus Function.
func NewSoftPlus[O DualValue](x O, beta, threshold O) *SoftPlus[O] {
	return &SoftPlus[O]{
		x:         x,
		beta:      beta,
		threshold: threshold,
	}
}

// Operands returns the list of operands.
func (r *SoftPlus[O]) Operands() []O {
	return []O{r.x, r.beta, r.threshold}
}

// Forward computes the output of the function.
func (r *SoftPlus[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().ApplyWithAlpha(
		softPlus,
		r.beta.Value().Scalar().F64(),
		r.threshold.Value().Scalar().F64(),
	), nil
}

// Backward computes the backward pass.
func (r *SoftPlus[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(
			softPlusDeriv,
			r.beta.Value().Scalar().F64(),
			r.threshold.Value().Scalar().F64(),
		)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
	return nil
}
