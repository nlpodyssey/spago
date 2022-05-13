// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Dot is an operator to perform the dot product over two matrices.
// y = x1 dot x2
type Dot[T mat.DType, O Operand[T]] struct {
	x1       O
	x2       O
	operands []O
}

// NewDot returns a new Dot Function.
func NewDot[T mat.DType, O Operand[T]](x1 O, x2 O) *Dot[T, O] {
	return &Dot[T, O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Operands returns the list of operands.
func (r *Dot[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Dot[T, O]) Forward() mat.Matrix[T] {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	if mat.IsVector(r.x1.Value()) && mat.IsVector(r.x2.Value()) {
		return r.x1.Value().DotUnitary(r.x2.Value())
	}

	prod := r.x1.Value().Prod(r.x2.Value())
	defer mat.ReleaseMatrix(prod)
	return prod.Sum()
}

// Backward computes the backward pass.
func (r *Dot[T, O]) Backward(gy mat.Matrix[T]) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	gys := gy.Scalar().Float64()
	if r.x1.RequiresGrad() {
		gx := r.x2.Value().ProdScalar(gys)
		defer mat.ReleaseMatrix(gx)
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gx := r.x1.Value().ProdScalar(gys)
		defer mat.ReleaseMatrix(gx)
		r.x2.AccGrad(gx)
	}
}
