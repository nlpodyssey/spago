// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// SwishB function: f(x) = x * sigmoid.
//
// Reference: "Searching for Activation Functions" by Ramachandran et al, 2017.
// (https://arxiv.org/pdf/1710.05941.pdf)
type SwishB[O mat.Tensor] struct {
	x    O
	beta O // scalar
}

// NewSwishB returns a new SwishB Function.
func NewSwishB[O mat.Tensor](x O, beta O) *SwishB[O] {
	return &SwishB[O]{
		x:    x,
		beta: beta,
	}
}

// Operands returns the list of operands.
func (r *SwishB[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x, r.beta}
}

// Forward computes the output of the function.
func (r *SwishB[O]) Forward() (mat.Tensor, error) {
	y := r.x.Value().(mat.Matrix).ApplyWithAlpha(swishB, r.beta.Value().Item().F64())
	return y, nil
}

// Backward computes the backward pass.
func (r *SwishB[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ApplyWithAlpha(swishBDeriv, r.beta.Value().Item().F64())
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	if r.beta.RequiresGrad() {
		gb := r.beta.Value().(mat.Matrix).ZerosLike()
		// FIXME: avoid casting to specific type
		for i, x := range r.x.Value().Data().F64() {
			deriv := swishBBetaDeriv(x, r.beta.Value().Item().F64())
			gyi := gy.(mat.Matrix).ScalarAt(i).F64()
			gb.AddScalarInPlace(deriv * gyi)
		}
		r.beta.AccGrad(gb)
	}
	return nil
}
