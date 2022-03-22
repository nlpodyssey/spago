// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Prod is an operator to perform element-wise product over two values.
type Prod[T mat.DType, O Operand[T]] struct {
	x1 O
	x2 O
}

// NewProd returns a new Prod Function.
func NewProd[T mat.DType, O Operand[T]](x1 O, x2 O) *Prod[T, O] {
	return &Prod[T, O]{x1: x1, x2: x2}
}

// Square is an operator to perform element-wise square.
type Square[T mat.DType, O Operand[T]] struct {
	*Prod[T, O]
}

// NewSquare returns a new Prod Function with both operands set to the given value x.
func NewSquare[T mat.DType, O Operand[T]](x O) *Square[T, O] {
	return &Square[T, O]{Prod: &Prod[T, O]{x1: x, x2: x}}
}

// Operands returns the list of operands.
func (r *Prod[T, O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the node.
func (r *Prod[T, O]) Forward() mat.Matrix[T] {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	return x1v.Prod(x2v)
}

// Backward computes the backward pass.
func (r *Prod[T, O]) Backward(gy mat.Matrix[T]) {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, gy) || mat.VectorsOfSameSize(x1v, gy)) &&
		!(mat.SameDims(x2v, gy) || mat.VectorsOfSameSize(x2v, gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		gx := r.x2.Value().Prod(gy)
		defer mat.ReleaseMatrix(gx)
		r.x1.PropagateGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gx := r.x1.Value().Prod(gy)
		defer mat.ReleaseMatrix(gx)
		r.x2.PropagateGrad(gx)
	}
}
