// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &AddScalar{}

// AddScalar is an operator to perform element-wise addition over two values.
type AddScalar struct {
	x1 Operand
	x2 Operand // scalar
}

// NewAddScalar returns a new AddScalar Function.
func NewAddScalar(x1, x2 Operand) *AddScalar {
	return &AddScalar{x1: x1, x2: x2}
}

// Forward computes the output of the function.
// It doesn't backward on the scalar value x2.
func (r *AddScalar) Forward() mat.Matrix {
	return r.x1.Value().AddScalar(r.x2.Value().Scalar())
}

// Backward computes the backward pass.
func (r *AddScalar) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy)
	}
	if r.x2.RequiresGrad() {
		gx := mat.NewScalar(gy.Sum())
		defer mat.ReleaseDense(gx)
		r.x2.PropagateGrad(gx)
	}
}
