// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// UnaryElementwise is a single-input element-wise function.
type UnaryElementwise[T mat.DType, O Operand[T]] struct {
	x        O
	operands []O
	f        func(i, j int, v float64) float64 // function
	df       func(i, j int, v float64) float64 // derivative
}

// Operands returns the list of operands.
func (r *UnaryElementwise[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of this node.
func (r *UnaryElementwise[T, O]) Forward() mat.Matrix {
	return r.x.Value().Apply(r.f)
}

// Backward computes the backward pass.
func (r *UnaryElementwise[T, O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().Apply(r.df)
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
