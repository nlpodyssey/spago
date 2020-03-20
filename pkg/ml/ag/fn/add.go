// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/saientist/spago/pkg/mat"

// Element-wise sum over two values.
// y = x1 + x2
type Add struct {
	x1 Operand
	x2 Operand
}

func NewAdd(x1, x2 Operand) *Add {
	return &Add{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Add) Forward() mat.Matrix {
	return r.x1.Value().Add(r.x2.Value())
}

func (r *Add) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy)
	}
	if r.x2.RequiresGrad() {
		r.x2.PropagateGrad(gy)
	}
}
