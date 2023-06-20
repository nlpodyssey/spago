// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Exp is an operator to perform element-wise base-e exponential function.
type Exp[O mat.Tensor] struct {
	x O
}

// NewExp returns a new Exp Function.
func NewExp[O mat.Tensor](x O) *Exp[O] {
	return &Exp[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (e *Exp[O]) Operands() []mat.Tensor {
	return []mat.Tensor{e.x}
}

// Forward computes the output of the function.
func (e *Exp[O]) Forward() (mat.Tensor, error) {
	return e.x.Value().(mat.Matrix).Exp(), nil
}

// Backward computes the backward pass.
func (e *Exp[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(e.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if e.x.RequiresGrad() {
		gx := e.x.Value().(mat.Matrix).Exp()
		gx.ProdInPlace(gy.(mat.Matrix))
		e.x.AccGrad(gx)
	}
	return nil
}
