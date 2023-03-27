// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ELU is an operator to perform the ELU activation function.
// ELU(x) = max(0,x) + min(0,α ∗ (exp(x) − 1))
type ELU[O DualValue] struct {
	x     O
	alpha O // scalar
}

// NewELU returns a new ELU Function.
func NewELU[O DualValue](x O, alpha O) *ELU[O] {
	return &ELU[O]{
		x:     x,
		alpha: alpha,
	}
}

// Operands returns the list of operands.
func (r *ELU[O]) Operands() []O {
	return []O{r.x, r.alpha}
}

// Forward computes the output of the function.
func (r *ELU[O]) Forward() (mat.Matrix, error) {
	y := r.x.Value().ApplyWithAlpha(elu, r.alpha.Value().Scalar().F64())
	return y, nil
}

// Backward computes the backward pass.
func (r *ELU[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(eluDeriv, r.alpha.Value().Scalar().F64())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
	return nil
}
