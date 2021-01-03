// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &ReduceSum{}

// ReduceSum is an operator to perform reduce-sum function.
type ReduceSum struct {
	x Operand
}

// NewReduceSum returns a new ReduceSum Function.
func NewReduceSum(x Operand) *ReduceSum {
	return &ReduceSum{x: x}
}

// Forward computes the output of this function.
func (r *ReduceSum) Forward() mat.Matrix {
	return mat.NewScalar(r.x.Value().Sum())
}

// Backward computes the backward pass.
func (r *ReduceSum) Backward(gy mat.Matrix) {
	if !gy.IsScalar() {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar())
		defer mat.ReleaseDense(gx)
		r.x.PropagateGrad(gx)
	}
}
