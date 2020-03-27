// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

// Single-input, element-wise function.
type UnaryElementwise struct {
	x  Operand
	f  func(i, j int, v float64) float64 // function
	df func(i, j int, v float64) float64 // derivative
}

// Forward computes the output of this node.
func (r *UnaryElementwise) Forward() mat.Matrix {
	y := r.x.Value().ZerosLike()
	y.Apply(r.f, r.x.Value())
	return y
}

func (r *UnaryElementwise) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		gx.Apply(r.df, r.x.Value())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
