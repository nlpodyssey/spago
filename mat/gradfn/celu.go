// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// CELU is an operator to perform the CELU activation.
// CELU(x) = max(0,x) + min(0,α ∗ (exp(x/α) − 1))
type CELU[O mat.Tensor] struct {
	x     O
	alpha O // scalar
}

// NewCELU returns a new CELU Function.
func NewCELU[O mat.Tensor](x O, alpha O) *CELU[O] {
	return &CELU[O]{
		x:     x,
		alpha: alpha,
	}
}

// Operands returns the list of operands.
func (r *CELU[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x, r.alpha}
}

// Forward computes the output of the function.
func (r *CELU[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).ApplyWithAlpha(celu, r.alpha.Value().Item().F64()), nil
}

// Backward computes the backward pass.
func (r *CELU[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value().(mat.Matrix), gy.(mat.Matrix)) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ApplyWithAlpha(celuDeriv, r.alpha.Value().Item().F64())
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
