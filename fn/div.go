// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Div is an operator to perform element-wise division over two values.
type Div[O DualValue] struct {
	x1 O
	x2 O
}

// NewDiv returns a new Div Function.
func NewDiv[O DualValue](x1 O, x2 O) *Div[O] {
	return &Div[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *Div[O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *Div[O]) Forward() (mat.Matrix, error) {
	return r.x1.Value().Div(r.x2.Value()), nil
}

// Backward computes the backward pass.
func (r *Div[O]) Backward(gy mat.Matrix) error {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !mat.SameDims(x1v, gy) || !mat.SameDims(x2v, gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		gx := gy.Div(r.x2.Value())
		defer mat.ReleaseMatrix(gx)
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		x2sq := r.x2.Value().Prod(r.x2.Value())
		defer mat.ReleaseMatrix(x2sq)
		gx := r.x1.Value().Prod(gy)
		defer mat.ReleaseMatrix(gx)
		gx.ProdScalarInPlace(-1)
		gx.DivInPlace(x2sq)
		r.x2.AccGrad(gx)
	}
	return nil
}
