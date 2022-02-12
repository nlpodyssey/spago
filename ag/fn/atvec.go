// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &AtVec[float32]{}

// AtVec is an operator to obtain the i-th value of a vector.
type AtVec[T mat.DType] struct {
	x Operand[T]
	i int
}

// NewAtVec returns a new AtVec Function.
func NewAtVec[T mat.DType](x Operand[T], i int) *AtVec[T] {
	return &AtVec[T]{x: x, i: i}
}

// Forward computes the output of the function.
func (r *AtVec[T]) Forward() mat.Matrix[T] {
	return mat.NewScalar(r.x.Value().AtVec(r.i))
}

// Backward computes the backward pass.
func (r *AtVec[T]) Backward(gy mat.Matrix[T]) {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(dx)
		dx.SetVec(r.i, gy.Scalar())
		r.x.PropagateGrad(dx)
	}
}
