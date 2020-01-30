// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"saientist.dev/spago/pkg/mat"
)

// Single-input, reduce sum function.
type ReduceSum struct {
	x Operand
}

func NewReduceSum(x Operand) *ReduceSum {
	return &ReduceSum{x: x}
}

// Forward computes the output of this function.
func (r *ReduceSum) Forward() mat.Matrix {
	return mat.NewScalar(r.x.Value().Sum())
}

func (r *ReduceSum) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		r.x.PropagateGrad(mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar()))
	}
}
