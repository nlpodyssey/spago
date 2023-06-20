// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// AddScalar is an operator to perform element-wise addition over two values.
type AddScalar[O mat.Tensor] struct {
	x1 O
	x2 O // scalar
}

// NewAddScalar returns a new AddScalar Function.
func NewAddScalar[O mat.Tensor](x1, x2 O) *AddScalar[O] {
	return &AddScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *AddScalar[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the function.
// It doesn't backward on the scalar value x2.
func (r *AddScalar[O]) Forward() (mat.Tensor, error) {
	return r.x1.Value().(mat.Matrix).AddScalar(r.x2.Value().Item().F64()), nil
}

// Backward computes the backward pass.
func (r *AddScalar[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x1.Value().(mat.Matrix), gy.(mat.Matrix)) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy)
	}
	if r.x2.RequiresGrad() {
		gx := gy.(mat.Matrix).Sum()
		r.x2.AccGrad(gx)
	}
	return nil
}
