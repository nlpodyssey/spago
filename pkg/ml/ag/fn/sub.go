// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"saientist.dev/spago/pkg/mat"
)

// Element-wise subtraction over two values.
type Sub struct {
	x1 Operand
	x2 Operand
}

func NewSub(x1, x2 Operand) *Sub {
	return &Sub{x1: x1, x2: x2}
}

// Forward computes the output of the node.
func (r *Sub) Forward() mat.Matrix {
	return r.x1.Value().Sub(r.x2.Value())
}

func (r *Sub) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy)
	}
	if r.x2.RequiresGrad() {
		r.x2.PropagateGrad(gy.ProdScalar(-1.0))
	}
}
