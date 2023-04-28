// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// SubScalar is an element-wise subtraction function with a scalar value.
type SubScalar[O DualValue] struct {
	x1 O
	x2 O // scalar
}

// NewSubScalar returns a new SubScalar Function.
func NewSubScalar[O DualValue](x1 O, x2 O) *SubScalar[O] {
	return &SubScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *SubScalar[O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the node.
func (r *SubScalar[O]) Forward() (mat.Matrix, error) {
	return r.x1.Value().SubScalar(r.x2.Value().Scalar().F64()), nil
}

// Backward computes the backward pass.
func (r *SubScalar[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x1.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy) // equals to gy.ProdScalar(1.0)
	}
	if r.x2.RequiresGrad() {
		neg := gy.ProdScalar(-1)
		gx := neg.Sum()
		r.x2.AccGrad(gx)
	}
	return nil
}
