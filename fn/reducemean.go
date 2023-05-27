// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ReduceMean is an operator to perform reduce-mean function.
type ReduceMean[O DualValue] struct {
	x O
}

// NewReduceMean returns a new ReduceMean Function.
func NewReduceMean[O DualValue](x O) *ReduceMean[O] {
	return &ReduceMean[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *ReduceMean[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of this node.
func (r *ReduceMean[O]) Forward() (mat.Matrix, error) {
	xv := r.x.Value()
	return xv.Sum().ProdScalarInPlace(1 / float64(xv.Size())), nil
}

// Backward computes the backward pass.
func (r *ReduceMean[O]) Backward(gy mat.Matrix) error {
	if !mat.IsScalar(gy) {
		return fmt.Errorf("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		x := r.x.Value()
		size := x.Size()
		v := gy.Scalar().F64() / float64(size)
		gx := x.NewMatrix(mat.WithShape(size), mat.WithBacking(mat.CreateInitializedSlice(size, v)))
		r.x.AccGrad(gx)
	}
	return nil
}
