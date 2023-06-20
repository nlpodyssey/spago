// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ELU is an operator to perform the ELU activation function.
// ELU(x) = max(0,x) + min(0,α ∗ (exp(x) − 1))
type ELU[O mat.Tensor] struct {
	x     O
	alpha O // scalar
}

// NewELU returns a new ELU Function.
func NewELU[O mat.Tensor](x O, alpha O) *ELU[O] {
	return &ELU[O]{
		x:     x,
		alpha: alpha,
	}
}

// Operands returns the list of operands.
func (r *ELU[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x, r.alpha}
}

// Forward computes the output of the function.
func (r *ELU[O]) Forward() (mat.Tensor, error) {
	y := r.x.Value().(mat.Matrix).ApplyWithAlpha(elu, r.alpha.Value().Item().F64())
	return y, nil
}

// Backward computes the backward pass.
func (r *ELU[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ApplyWithAlpha(eluDeriv, r.alpha.Value().Item().F64())
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
