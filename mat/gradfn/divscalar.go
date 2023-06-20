// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/nlpodyssey/spago/mat"
)

// DivScalar is an operator to perform element-wise division with a scalar value.
type DivScalar[O mat.Tensor] struct {
	x1 O
	x2 O // scalar
}

// NewDivScalar returns a new DivScalar Function.
func NewDivScalar[O mat.Tensor](x1 O, x2 O) *DivScalar[O] {
	return &DivScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *DivScalar[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *DivScalar[O]) Forward() (mat.Tensor, error) {
	return r.x1.Value().(mat.Matrix).ProdScalar(1.0 / r.x2.Value().Item().F64()), nil
}

// Backward computes the backward pass.
func (r *DivScalar[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x1.Value().(mat.Matrix), gy.(mat.Matrix)) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy.(mat.Matrix).ProdScalar(1.0 / r.x2.Value().Item().F64()))
	}
	if r.x2.RequiresGrad() {
		x2 := r.x2.Value().Item().F64()

		a := r.x1.Value().(mat.Matrix).ProdScalar(1 / -(x2 * x2))
		b := gy.(mat.Matrix).Prod(a)
		gx := b.Sum()
		r.x2.AccGrad(gx)
	}
	return nil
}
