// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// SwishB function: f(x) = x * sigmoid.
//
// Reference: "Searching for Activation Functions" by Ramachandran et al, 2017.
// (https://arxiv.org/pdf/1710.05941.pdf)
type SwishB[O DualValue] struct {
	x    O
	beta O // scalar
}

// NewSwishB returns a new SwishB Function.
func NewSwishB[O DualValue](x O, beta O) *SwishB[O] {
	return &SwishB[O]{
		x:    x,
		beta: beta,
	}
}

// Operands returns the list of operands.
func (r *SwishB[O]) Operands() []O {
	return []O{r.x, r.beta}
}

// Forward computes the output of the function.
func (r *SwishB[O]) Forward() (mat.Matrix, error) {
	y := r.x.Value().ApplyWithAlpha(swishB, r.beta.Value().Scalar().F64())
	return y, nil
}

// Backward computes the backward pass.
func (r *SwishB[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(swishBDeriv, r.beta.Value().Scalar().F64())
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
	if r.beta.RequiresGrad() {
		gb := r.beta.Value().ZerosLike()
		// FIXME: avoid casting to specific type
		for i, x := range r.x.Value().Data().F64() {
			deriv := swishBBetaDeriv(x, r.beta.Value().Scalar().F64())
			gyi := gy.ScalarAt(i).F64()
			gb.AddScalarInPlace(deriv * gyi)
		}
		r.beta.AccGrad(gb)
	}
	return nil
}
