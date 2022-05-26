// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Sub is an element-wise subtraction function over two values.
type Sub[O Operand] struct {
	x1       O
	x2       O
	operands []O
}

// NewSub returns a new Sub Function.
func NewSub[O Operand](x1 O, x2 O) *Sub[O] {
	return &Sub[O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Operands returns the list of operands.
func (r *Sub[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the node.
func (r *Sub[O]) Forward() mat.Matrix {
	return r.x1.Value().Sub(r.x2.Value())
}

// Backward computes the backward pass.
func (r *Sub[O]) Backward(gy mat.Matrix) {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !mat.SameDims(x1v, gy) || !mat.SameDims(x2v, gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy)
	}
	if r.x2.RequiresGrad() {
		gx := gy.ProdScalar(-1.0)
		defer mat.ReleaseMatrix(gx)
		r.x2.AccGrad(gx)
	}
}
