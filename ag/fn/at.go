// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &At[float32]{}

// At is an operator to obtain the i,j-th value of a matrix.
type At[T mat.DType] struct {
	x Operand[T]
	i int
	j int
}

// NewAt returns a new At Function.
func NewAt[T mat.DType](x Operand[T], i int, j int) *At[T] {
	return &At[T]{x: x, i: i, j: j}
}

// Forward computes the output of the function.
func (r *At[T]) Forward() mat.Matrix[T] {
	return mat.NewScalar(r.x.Value().At(r.i, r.j))
}

// Backward computes the backward pass.
func (r *At[T]) Backward(gy mat.Matrix[T]) {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(dx)
		dx.Set(r.i, r.j, gy.Scalar())
		r.x.PropagateGrad(dx)
	}
}
