// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ReduceMax is an operator to perform reduce-max function.
// It gets the maximum element of the Operand x
type ReduceMax[O Operand] struct {
	x      O
	argmax int
}

// NewReduceMax returns a new ReduceMax Function.
func NewReduceMax[O Operand](x O) *ReduceMax[O] {
	return &ReduceMax[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *ReduceMax[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of this function.
func (r *ReduceMax[O]) Forward() mat.Matrix {
	xv := r.x.Value()
	r.argmax = xv.ArgMax()
	return xv.AtVec(r.argmax)
}

// Backward computes the backward pass.
func (r *ReduceMax[O]) Backward(gy mat.Matrix) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		x := r.x.Value()
		gx := x.ZerosLike()
		defer mat.ReleaseMatrix(gx)
		gx.SetVec(r.argmax, gy)
		r.x.AccGrad(gx)
	}
}
