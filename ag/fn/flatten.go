// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &Flatten[float32]{}

// Flatten is a Function to reshape a matrix-operand into a "flattened" row vector.
type Flatten[T mat.DType] struct {
	x Operand[T]
}

// NewFlatten returns a new Flatten Function.
func NewFlatten[T mat.DType](x Operand[T]) *Flatten[T] {
	return &Flatten[T]{x: x}
}

// Forward computes the output of the node.
func (r *Flatten[T]) Forward() mat.Matrix[T] {
	return r.x.Value().Flatten()
}

// Backward computes the backward pass.
func (r *Flatten[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.IsVector(gy) && r.x.Value().Size() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
