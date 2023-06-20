// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Div is an operator to perform element-wise division over two values.
type Div[O mat.Tensor] struct {
	x1 O
	x2 O
}

// NewDiv returns a new Div Function.
func NewDiv[O mat.Tensor](x1 O, x2 O) *Div[O] {
	return &Div[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *Div[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *Div[O]) Forward() (mat.Tensor, error) {
	return r.x1.Value().(mat.Matrix).Div(r.x2.Value().(mat.Matrix)), nil
}

// Backward computes the backward pass.
func (r *Div[O]) Backward(gy mat.Tensor) error {
	x1v := r.x1.Value().(mat.Matrix)
	x2v := r.x2.Value().(mat.Matrix)
	if !mat.SameDims(x1v, gy.(mat.Matrix)) || !mat.SameDims(x2v, gy.(mat.Matrix)) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		gx := gy.(mat.Matrix).Div(r.x2.Value().(mat.Matrix))
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		x2sq := r.x2.Value().(mat.Matrix).Prod(r.x2.Value().(mat.Matrix))
		gx := r.x1.Value().(mat.Matrix).Prod(gy.(mat.Matrix))
		gx.ProdScalarInPlace(-1)
		gx.DivInPlace(x2sq)
		r.x2.AccGrad(gx)
	}
	return nil
}
