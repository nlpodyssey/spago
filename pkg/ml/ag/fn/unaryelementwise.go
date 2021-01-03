// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &UnaryElementwise{}

// UnaryElementwise is a single-input element-wise function.
type UnaryElementwise struct {
	x  Operand
	f  func(i, j int, v mat.Float) mat.Float // function
	df func(i, j int, v mat.Float) mat.Float // derivative
}

// Forward computes the output of this node.
func (r *UnaryElementwise) Forward() mat.Matrix {
	y := mat.GetDenseWorkspace(r.x.Value().Dims())
	y.Apply(r.f, r.x.Value())
	return y
}

// Backward computes the backward pass.
func (r *UnaryElementwise) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDenseWorkspace(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.Apply(r.df, r.x.Value())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
