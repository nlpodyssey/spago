// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// CELU is an operator to perform the CELU activation.
// CELU(x) = max(0,x) + min(0,α ∗ (exp(x/α) − 1))
type CELU[O DualValue] struct {
	x     O
	alpha O // scalar
}

// NewCELU returns a new CELU Function.
func NewCELU[O DualValue](x O, alpha O) *CELU[O] {
	return &CELU[O]{
		x:     x,
		alpha: alpha,
	}
}

// Operands returns the list of operands.
func (r *CELU[O]) Operands() []O {
	return []O{r.x, r.alpha}
}

// Forward computes the output of the function.
func (r *CELU[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().ApplyWithAlpha(celu, r.alpha.Value().Scalar().F64()), nil
}

// Backward computes the backward pass.
func (r *CELU[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(celuDeriv, r.alpha.Value().Scalar().F64())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
	return nil
}
