// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Copy is an operator to perform copy function.
// y = x
type Copy[O DualValue] struct {
	x O
}

// NewCopy returns a new Copy Function.
func NewCopy[O DualValue](x O) *Copy[O] {
	return &Copy[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Copy[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *Copy[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().Clone(), nil
}

// Backward computes the backward pass.
func (r *Copy[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	r.x.AccGrad(gy)
	return nil
}
