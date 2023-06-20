// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ReverseSubScalar is the element-wise subtraction function over two values.
type ReverseSubScalar[O mat.Tensor] struct {
	x1 O
	x2 O // scalar
}

// NewReverseSubScalar returns a new ReverseSubScalar Function.
func NewReverseSubScalar[O mat.Tensor](x1 O, x2 O) *ReverseSubScalar[O] {
	return &ReverseSubScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *ReverseSubScalar[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *ReverseSubScalar[O]) Forward() (mat.Tensor, error) {
	x1 := r.x1.Value()
	x2 := r.x2.Value()
	return x1.(mat.Matrix).ProdScalar(-1).AddScalarInPlace(x2.Item().F64()), nil
}

// Backward computes the backward pass.
func (r *ReverseSubScalar[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x1.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		gx := gy.(mat.Matrix).ProdScalar(-1)
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gx := gy.(mat.Matrix).Sum()
		r.x2.AccGrad(gx)
	}
	return nil
}
