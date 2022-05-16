// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// ReduceMean is an operator to perform reduce-mean function.
type ReduceMean[O Operand] struct {
	x        O
	operands []O
}

// NewReduceMean returns a new ReduceMean Function.
func NewReduceMean[O Operand](x O) *ReduceMean[O] {
	return &ReduceMean[O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *ReduceMean[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of this node.
func (r *ReduceMean[O]) Forward() mat.Matrix {
	xv := r.x.Value()
	return xv.Sum().ProdScalarInPlace(1 / float64(xv.Size()))
}

// Backward computes the backward pass.
func (r *ReduceMean[O]) Backward(gy mat.Matrix) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		x := r.x.Value()
		size := x.Size()
		v := gy.Scalar().F64() / float64(size)
		gx := x.NewInitVec(size, v)
		defer mat.ReleaseMatrix(gx)
		r.x.AccGrad(gx)
	}
}
