// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Flatten is a Function to reshape a matrix-operand into a "flattened" row vector.
type Flatten[T mat.DType, O Operand[T]] struct {
	x        O
	operands []O
}

// NewFlatten returns a new Flatten Function.
func NewFlatten[T mat.DType, O Operand[T]](x O) *Flatten[T, O] {
	return &Flatten[T, O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *Flatten[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the node.
func (r *Flatten[T, O]) Forward() mat.Matrix[T] {
	return r.x.Value().Flatten()
}

// Backward computes the backward pass.
func (r *Flatten[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.IsVector(gy) && r.x.Value().Size() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
