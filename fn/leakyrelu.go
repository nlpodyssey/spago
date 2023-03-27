// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// LeakyReLU is an operator to perform the LeakyReLU activation function.
// LeakyReLU(x) = max(0,x) + slope ° min(0,x)
type LeakyReLU[O DualValue] struct {
	x     O
	alpha O // scalar
}

// NewLeakyReLU returns a new LeakyReLU Function.
func NewLeakyReLU[O DualValue](x, alpha O) *LeakyReLU[O] {
	return &LeakyReLU[O]{
		x:     x,
		alpha: alpha,
	}
}

// Operands returns the list of operands.
func (r *LeakyReLU[O]) Operands() []O {
	return []O{r.x, r.alpha}
}

// Forward computes the output of the function.
func (r *LeakyReLU[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().ApplyWithAlpha(leakyReLU, r.alpha.Value().Scalar().F64()), nil
}

// Backward computes the backward pass.
func (r *LeakyReLU[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(leakyReLUDeriv, r.alpha.Value().Scalar().F64())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
	return nil
}
