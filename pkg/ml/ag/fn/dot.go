// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import mat "github.com/nlpodyssey/spago/pkg/mat32"

var _ Function = &Dot{}

// Dot is an operator to perform the dot product over two matrices.
// y = x1 dot x2
type Dot struct {
	x1 Operand
	x2 Operand
}

// NewDot returns a new Dot Function.
func NewDot(x1, x2 Operand) *Dot {
	return &Dot{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Dot) Forward() mat.Matrix {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	var y mat.Float = 0.0
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

// Backward computes the backward pass.
func (r *Dot) Backward(gy mat.Matrix) {
	if !gy.IsScalar() {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x1.RequiresGrad() {
		dx := mat.GetDenseWorkspace(r.x1.Value().Dims())
		defer mat.ReleaseDense(dx)
		for i := 0; i < r.x1.Value().Rows(); i++ {
			for j := 0; j < r.x1.Value().Columns(); j++ {
				dx.Set(i, j, gy.Scalar()*r.x2.Value().At(i, j))
			}
		}
		r.x1.PropagateGrad(dx)
	}
	if r.x2.RequiresGrad() {
		dx := mat.GetDenseWorkspace(r.x2.Value().Dims())
		defer mat.ReleaseDense(dx)
		for i := 0; i < r.x2.Value().Rows(); i++ {
			for j := 0; j < r.x2.Value().Columns(); j++ {
				dx.Set(i, j, gy.Scalar()*r.x1.Value().At(i, j))
			}
		}
		r.x2.PropagateGrad(dx)
	}
}
