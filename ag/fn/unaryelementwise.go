// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &UnaryElementwise[float32]{}

// UnaryElementwise is a single-input element-wise function.
type UnaryElementwise[T mat.DType] struct {
	x  Operand[T]
	f  func(i, j int, v T) T // function
	df func(i, j int, v T) T // derivative
}

// Forward computes the output of this node.
func (r *UnaryElementwise[T]) Forward() mat.Matrix[T] {
	return r.x.Value().Apply(r.f)
}

// Backward computes the backward pass.
func (r *UnaryElementwise[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().Apply(r.df)
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
