// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &Div[float32]{}

// Div is an operator to perform element-wise division over two values.
type Div[T mat.DType] struct {
	x1 Operand[T]
	x2 Operand[T]
}

// NewDiv returns a new Div Function.
func NewDiv[T mat.DType](x1, x2 Operand[T]) *Div[T] {
	return &Div[T]{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Div[T]) Forward() mat.Matrix[T] {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	return x1v.Div(x2v)
}

// Backward computes the backward pass.
func (r *Div[T]) Backward(gy mat.Matrix[T]) {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, gy) || mat.VectorsOfSameSize(x1v, gy)) &&
		!(mat.SameDims(x2v, gy) || mat.VectorsOfSameSize(x2v, gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		gx := gy.Div(r.x2.Value())
		defer mat.ReleaseMatrix(gx)
		r.x1.PropagateGrad(gx)
	}
	if r.x2.RequiresGrad() {
		x2sq := r.x2.Value().Prod(r.x2.Value())
		defer mat.ReleaseMatrix(x2sq)
		gx := r.x1.Value().Prod(gy)
		defer mat.ReleaseMatrix(gx)
		gx.ProdScalarInPlace(-1)
		gx.DivInPlace(x2sq)
		r.x2.PropagateGrad(gx)
	}
}
