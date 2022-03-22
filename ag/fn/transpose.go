// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Transpose is a Function to calculate the transpose of the matrix-operand.
type Transpose[T mat.DType, O Operand[T]] struct {
	x O
}

// NewTranspose returns a new Transpose Function.
func NewTranspose[T mat.DType, O Operand[T]](x O) *Transpose[T, O] {
	return &Transpose[T, O]{x: x}
}

// Operands returns the list of operands.
func (r *Transpose[T, O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the node.
func (r *Transpose[T, O]) Forward() mat.Matrix[T] {
	return r.x.Value().T()
}

// Backward computes the backward pass.
func (r *Transpose[T, O]) Backward(gy mat.Matrix[T]) {
	if r.x.Value().Columns() != gy.Rows() && r.x.Value().Rows() != gy.Columns() {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.T()
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
