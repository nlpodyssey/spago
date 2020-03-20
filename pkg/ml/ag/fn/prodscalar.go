// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/saientist/spago/pkg/mat"
)

// The element-wise product with a scalar value.
type ProdScalar struct {
	x1 Operand
	x2 Operand // scalar
}

func NewProdScalar(x1, x2 Operand) *ProdScalar {
	return &ProdScalar{x1: x1, x2: x2}
}

// Forward computes the output of the node.
func (r *ProdScalar) Forward() mat.Matrix {
	return r.x1.Value().ProdScalar(r.x2.Value().Scalar())
}

func (r *ProdScalar) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy.ProdScalar(r.x2.Value().Scalar()))
	}
	if r.x2.RequiresGrad() {
		gx := 0.0
		for i := 0; i < gy.Rows(); i++ {
			for j := 0; j < gy.Columns(); j++ {
				gx += gy.At(i, j) * r.x1.Value().At(i, j)
			}
		}
		r.x2.PropagateGrad(mat.NewScalar(gx))
	}
}
