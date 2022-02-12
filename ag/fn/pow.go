// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &Pow[float32]{}

// Pow is an operator to perform element-wise pow function.
type Pow[T mat.DType] struct {
	x     Operand[T]
	power T
}

// NewPow returns a new Pow Function.
func NewPow[T mat.DType](x Operand[T], power T) *Pow[T] {
	return &Pow[T]{x: x, power: power}
}

// Forward computes the output of the function.
func (r *Pow[T]) Forward() mat.Matrix[T] {
	return r.x.Value().Pow(r.power)
}

// Backward computes the backward pass.
func (r *Pow[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().Pow(r.power - 1)
		defer mat.ReleaseMatrix(gx)
		gx.ProdScalarInPlace(r.power).ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
