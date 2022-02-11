// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &Vec[float32]{}

// Vec is a Function to reshape an matrix-operand into a column vector.
type Vec[T mat.DType] struct {
	x Operand[T]
}

// NewVec returns a new Vec Function.
func NewVec[T mat.DType](x Operand[T]) *Vec[T] {
	return &Vec[T]{x: x}
}

// Forward computes the output of the node.
func (r *Vec[T]) Forward() mat.Matrix[T] {
	return r.x.Value().Reshape(r.x.Value().Size(), 1)
}

// Backward computes the backward pass.
func (r *Vec[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.IsVector(gy) && r.x.Value().Size() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
