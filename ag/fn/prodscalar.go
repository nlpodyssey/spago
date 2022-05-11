// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ProdScalar is an operator to perform element-wise product with a scalar value.
type ProdScalar[T mat.DType, O Operand[T]] struct {
	x1       O
	x2       O // scalar
	operands []O
}

// NewProdScalar returns a new ProdScalar Function.
func NewProdScalar[T mat.DType, O Operand[T]](x1 O, x2 O) *ProdScalar[T, O] {
	return &ProdScalar[T, O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Operands returns the list of operands.
func (r *ProdScalar[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the node.
func (r *ProdScalar[T, O]) Forward() mat.Matrix[T] {
	return r.x1.Value().ProdScalar(float64(r.x2.Value().Scalar()))
}

// Backward computes the backward pass.
func (r *ProdScalar[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		gx := gy.ProdScalar(float64(r.x2.Value().Scalar()))
		defer mat.ReleaseMatrix(gx)
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		var gx T = 0.0
		for i := 0; i < gy.Rows(); i++ {
			for j := 0; j < gy.Columns(); j++ {
				gx += gy.At(i, j) * r.x1.Value().At(i, j)
			}
		}
		scalar := mat.NewScalar(gx)
		defer mat.ReleaseDense(scalar)
		r.x2.AccGrad(scalar)
	}
}
