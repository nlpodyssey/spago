// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Threshold function: f(x) = x if x > threshold; k otherwise.
type Threshold[O mat.Tensor] struct {
	x         O
	threshold O // scalar
	k         O // scalar
}

// NewThreshold returns a new Threshold Function.
func NewThreshold[O mat.Tensor](x O, threshold, k O) *Threshold[O] {
	return &Threshold[O]{
		x:         x,
		threshold: threshold,
		k:         k,
	}
}

// Operands returns the list of operands.
func (r *Threshold[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x, r.threshold, r.k}
}

// Forward computes the output of the function.
func (r *Threshold[O]) Forward() (mat.Tensor, error) {
	y := r.x.Value().(mat.Matrix).ApplyWithAlpha(
		threshold,
		r.threshold.Value().Item().F64(),
		r.k.Value().Item().F64(),
	)
	return y, nil
}

// Backward computes the backward pass.
func (r *Threshold[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ApplyWithAlpha(
			thresholdDeriv,
			r.threshold.Value().Item().F64(),
			r.k.Value().Item().F64(),
		)
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
