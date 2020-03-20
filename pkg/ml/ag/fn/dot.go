// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/saientist/spago/pkg/mat"

// Dot product over two matrices.
// y = x1 dot x2
type Dot struct {
	x1 Operand
	x2 Operand
}

func NewDot(x1, x2 Operand) *Dot {
	return &Dot{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Dot) Forward() mat.Matrix {
	y := 0.0
	if r.x1.Value().IsVector() && r.x2.Value().IsVector() {
		y = r.x1.Value().DotUnitary(r.x2.Value())
	} else {
		for i := 0; i < r.x1.Value().Rows(); i++ {
			for j := 0; j < r.x1.Value().Columns(); j++ {
				y += r.x1.Value().At(i, j) * r.x2.Value().At(i, j)
			}
		}
	}
	return mat.NewScalar(y)
}

func (r *Dot) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		dx := mat.NewEmptyDense(r.x1.Value().Dims())
		for i := 0; i < r.x1.Value().Rows(); i++ {
			for j := 0; j < r.x1.Value().Columns(); j++ {
				dx.Set(gy.Scalar()*r.x2.Value().At(i, j), i, j)
			}
		}
		r.x1.PropagateGrad(dx)
	}
	if r.x2.RequiresGrad() {
		dx := mat.NewEmptyDense(r.x2.Value().Dims())
		for i := 0; i < r.x2.Value().Rows(); i++ {
			for j := 0; j < r.x2.Value().Columns(); j++ {
				dx.Set(gy.Scalar()*r.x1.Value().At(i, j), i, j)
			}
		}
		r.x2.PropagateGrad(dx)
	}
}
