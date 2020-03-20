// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/saientist/spago/pkg/mat"
)

// Element-wise product over two values.
type Prod struct {
	x1 Operand
	x2 Operand
}

func NewProd(x1, x2 Operand) *Prod {
	return &Prod{x1: x1, x2: x2}
}

func NewSquare(x Operand) *Prod {
	return &Prod{x1: x, x2: x}
}

// Forward computes the output of the node.
func (r *Prod) Forward() mat.Matrix {
	return r.x1.Value().Prod(r.x2.Value())
}

func (r *Prod) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(r.x2.Value().Prod(gy))
	}
	if r.x2.RequiresGrad() {
		r.x2.PropagateGrad(r.x1.Value().Prod(gy))
	}
}
