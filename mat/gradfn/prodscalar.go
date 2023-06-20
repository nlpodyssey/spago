// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ProdScalar is an operator to perform element-wise product with a scalar value.
type ProdScalar[O mat.Tensor] struct {
	x1 O
	x2 O // scalar
}

// NewProdScalar returns a new ProdScalar Function.
func NewProdScalar[O mat.Tensor](x1 O, x2 O) *ProdScalar[O] {
	return &ProdScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *ProdScalar[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the node.
func (r *ProdScalar[O]) Forward() (mat.Tensor, error) {
	return r.x1.Value().(mat.Matrix).ProdScalar(r.x2.Value().Item().F64()), nil
}

// Backward computes the backward pass.
func (r *ProdScalar[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x1.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		gx := gy.(mat.Matrix).ProdScalar(r.x2.Value().Item().F64())
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		prod := gy.(mat.Matrix).Prod(r.x1.Value().(mat.Matrix))
		gx := prod.Sum()
		r.x2.AccGrad(gx)
	}
	return nil
}
