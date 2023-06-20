// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Transpose is a Function to calculate the transpose of the matrix-operand.
type Transpose[O mat.Tensor] struct {
	x O
}

// NewTranspose returns a new Transpose Function.
func NewTranspose[O mat.Tensor](x O) *Transpose[O] {
	return &Transpose[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Transpose[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the node.
func (r *Transpose[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).T(), nil
}

// Backward computes the backward pass.
func (r *Transpose[O]) Backward(gy mat.Tensor) error {
	if r.x.Value().Shape()[1] != gy.Shape()[0] && r.x.Value().Shape()[0] != gy.Shape()[1] {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := gy.(mat.Matrix).T()
		r.x.AccGrad(gx)
	}
	return nil
}
