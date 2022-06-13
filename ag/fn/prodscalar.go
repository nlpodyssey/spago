// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ProdScalar is an operator to perform element-wise product with a scalar value.
type ProdScalar[O Operand] struct {
	x1 O
	x2 O // scalar
}

// NewProdScalar returns a new ProdScalar Function.
func NewProdScalar[O Operand](x1 O, x2 O) *ProdScalar[O] {
	return &ProdScalar[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *ProdScalar[O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the node.
func (r *ProdScalar[O]) Forward() mat.Matrix {
	return r.x1.Value().ProdScalar(r.x2.Value().Scalar().F64())
}

// Backward computes the backward pass.
func (r *ProdScalar[O]) Backward(gy mat.Matrix) {
	if !mat.SameDims(r.x1.Value(), gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		gx := gy.ProdScalar(r.x2.Value().Scalar().F64())
		defer mat.ReleaseMatrix(gx)
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		prod := gy.Prod(r.x1.Value())
		defer mat.ReleaseMatrix(prod)
		gx := prod.Sum()
		defer mat.ReleaseMatrix(gx)
		r.x2.AccGrad(gx)
	}
}
