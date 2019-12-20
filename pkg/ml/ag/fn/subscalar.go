// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
)

// Element-wise subtraction with a scalar value.
type SubScalar struct {
	x1 Operand
	x2 Operand // scalar
}

func NewSubScalar(x1, x2 Operand) *SubScalar {
	return &SubScalar{x1: x1, x2: x2}
}

// Forward computes the output of the node.
func (r *SubScalar) Forward() mat.Matrix {
	return r.x1.Value().SubScalar(r.x2.Value().Scalar())
}

func (r *SubScalar) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy) // equals to gy.ProdScalar(1.0)
	}
}
