// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &Transpose[float32]{}

// Transpose is a Function to calculate the transpose of the matrix-operand.
type Transpose[T mat.DType] struct {
	x Operand[T]
}

// NewTranspose returns a new Transpose Function.
func NewTranspose[T mat.DType](x Operand[T]) *Transpose[T] {
	return &Transpose[T]{x: x}
}

// Forward computes the output of the node.
func (r *Transpose[T]) Forward() mat.Matrix[T] {
	return r.x.Value().T()
}

// Backward computes the backward pass.
func (r *Transpose[T]) Backward(gy mat.Matrix[T]) {
	if r.x.Value().Columns() != gy.Rows() && r.x.Value().Rows() != gy.Columns() {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.T()
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
