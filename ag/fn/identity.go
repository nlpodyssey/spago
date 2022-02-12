// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

var _ Function[float32] = &Identity[float32]{}

// Identity is an operator to perform identity function.
// y = x
type Identity[T mat.DType] struct {
	x Operand[T]
}

// NewIdentity returns a new Identity Function.
func NewIdentity[T mat.DType](x Operand[T]) *Identity[T] {
	return &Identity[T]{x: x}
}

// Forward computes the output of the function.
func (r *Identity[T]) Forward() mat.Matrix[T] {
	return r.x.Value().Clone()
}

// Backward computes the backward pass.
func (r *Identity[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	r.x.PropagateGrad(gy)
}
