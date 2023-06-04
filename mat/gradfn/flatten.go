// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Flatten is a Function to reshape a matrix-operand into a "flattened" row vector.
type Flatten[O DualValue] struct {
	x O
}

// NewFlatten returns a new Flatten Function.
func NewFlatten[O DualValue](x O) *Flatten[O] {
	return &Flatten[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Flatten[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the node.
func (r *Flatten[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().Flatten(), nil
}

// Backward computes the backward pass.
func (r *Flatten[O]) Backward(gy mat.Matrix) error {
	if !(mat.IsVector(gy) && r.x.Value().Size() == gy.Size()) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Shape()...)
		r.x.AccGrad(gx)
	}
	return nil
}
