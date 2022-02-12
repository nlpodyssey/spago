// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &ReduceMean[float32]{}

// ReduceMean is an operator to perform reduce-mean function.
type ReduceMean[T mat.DType] struct {
	x Operand[T]
}

// NewReduceMean returns a new ReduceMean Function.
func NewReduceMean[T mat.DType](x Operand[T]) *ReduceMean[T] {
	return &ReduceMean[T]{x: x}
}

// Forward computes the output of this node.
func (r *ReduceMean[T]) Forward() mat.Matrix[T] {
	return mat.NewScalar(r.x.Value().Sum() / T(r.x.Value().Size()))
}

// Backward computes the backward pass.
func (r *ReduceMean[T]) Backward(gy mat.Matrix[T]) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar()/T(r.x.Value().Size()))
		defer mat.ReleaseDense(gx)
		r.x.PropagateGrad(gx)
	}
}
