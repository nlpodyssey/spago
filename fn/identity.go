// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Identity is an operator to perform identity function.
// y = x
type Identity[O DualValue] struct {
	x O
}

// NewIdentity returns a new Identity Function.
func NewIdentity[O DualValue](x O) *Identity[O] {
	return &Identity[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Identity[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *Identity[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().Clone(), nil
}

// Backward computes the backward pass.
func (r *Identity[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	r.x.AccGrad(gy)
	return nil
}
