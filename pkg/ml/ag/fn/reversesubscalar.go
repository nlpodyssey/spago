// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "saientist.dev/spago/pkg/mat"

// Element-wise subtraction over two values.
type ReverseSubScalar struct {
	x1 Operand
	x2 Operand // scalar
}

func NewReverseSubScalar(x1, x2 Operand) *ReverseSubScalar {
	return &ReverseSubScalar{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *ReverseSubScalar) Forward() mat.Matrix {
	return mat.NewInitDense(r.x1.Value().Rows(), r.x1.Value().Columns(), r.x2.Value().Scalar()).Sub(r.x1.Value())
}

func (r *ReverseSubScalar) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy.ProdScalar(-1.0))
	}
}
