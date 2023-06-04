// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// SoftShrink function: f(x) = x − λ if x > λ; x + λ if x < −λ; 0 otherwise.
type SoftShrink[O DualValue] struct {
	x      O
	lambda O // scalar
}

// NewSoftShrink returns a new SoftShrink Function.
func NewSoftShrink[O DualValue](x O, lambda O) *SoftShrink[O] {
	return &SoftShrink[O]{
		x:      x,
		lambda: lambda,
	}
}

// Operands returns the list of operands.
func (r *SoftShrink[O]) Operands() []O {
	return []O{r.x, r.lambda}
}

// Forward computes the output of the function.
func (r *SoftShrink[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().ApplyWithAlpha(softShrink, r.lambda.Value().Scalar().F64()), nil
}

// Backward computes the backward pass.
func (r *SoftShrink[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(softShrinkDeriv, r.lambda.Value().Scalar().F64())
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
	return nil
}