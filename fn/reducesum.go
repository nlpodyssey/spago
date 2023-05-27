// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ReduceSum is an operator to perform reduce-sum function.
type ReduceSum[O DualValue] struct {
	x O
}

// NewReduceSum returns a new ReduceSum Function.
func NewReduceSum[O DualValue](x O) *ReduceSum[O] {
	return &ReduceSum[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *ReduceSum[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of this function.
func (r *ReduceSum[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().Sum(), nil
}

// Backward computes the backward pass.
func (r *ReduceSum[O]) Backward(gy mat.Matrix) error {
	if !mat.IsScalar(gy) {
		return fmt.Errorf("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		x := r.x.Value()
		gx := x.NewMatrix(mat.WithShape(x.Size()), mat.WithBacking(mat.CreateInitializedSlice(x.Size(), gy.Scalar().F64())))
		r.x.AccGrad(gx)
	}
	return nil
}
