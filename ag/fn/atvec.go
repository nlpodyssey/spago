// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// AtVec is an operator to obtain the i-th value of a vector.
type AtVec[T mat.DType, O Operand[T]] struct {
	x O
	i int
}

// NewAtVec returns a new AtVec Function.
func NewAtVec[T mat.DType, O Operand[T]](x O, i int) *AtVec[T, O] {
	return &AtVec[T, O]{x: x, i: i}
}

// Operands returns the list of operands.
func (r *AtVec[T, O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *AtVec[T, O]) Forward() mat.Matrix[T] {
	return mat.NewScalar(r.x.Value().AtVec(r.i))
}

// Backward computes the backward pass.
func (r *AtVec[T, O]) Backward(gy mat.Matrix[T]) {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(dx)
		dx.SetVec(r.i, gy.Scalar())
		r.x.PropagateGrad(dx)
	}
}
