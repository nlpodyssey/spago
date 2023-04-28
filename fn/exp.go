// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Exp is an operator to perform element-wise base-e exponential function.
type Exp[O DualValue] struct {
	x O
}

// NewExp returns a new Exp Function.
func NewExp[O DualValue](x O) *Exp[O] {
	return &Exp[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (e *Exp[O]) Operands() []O {
	return []O{e.x}
}

// Forward computes the output of the function.
func (e *Exp[O]) Forward() (mat.Matrix, error) {
	return e.x.Value().Exp(), nil
}

// Backward computes the backward pass.
func (e *Exp[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(e.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if e.x.RequiresGrad() {
		gx := e.x.Value().Exp()
		gx.ProdInPlace(gy)
		e.x.AccGrad(gx)
	}
	return nil
}
