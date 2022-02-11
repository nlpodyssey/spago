// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

var _ Function[float32] = &RotateR[float32]{}

// RotateR is a function to perform a right circular shift of a vector.
type RotateR[T mat.DType] struct {
	x Operand[T]
	i int
}

// NewRotateR returns a new RotateR Function. `i` is the number of places by
// which the elements are shifted.
func NewRotateR[T mat.DType](x Operand[T], i int) *RotateR[T] {
	return &RotateR[T]{x: x, i: i}
}

// Forward computes the output of the function.
func (r *RotateR[T]) Forward() mat.Matrix[T] {
	xv := r.x.Value().Data()
	return mat.NewVecDense(rotateR(xv, r.i))
}

// Backward computes the backward pass.
func (r *RotateR[T]) Backward(gy mat.Matrix[T]) {
	if r.x.RequiresGrad() {
		gx := mat.NewVecDense(rotateL(gy.Data(), r.i))
		defer mat.ReleaseDense(gx)
		r.x.PropagateGrad(gx)
	}
}

func rotateR[T mat.DType](a []T, i int) []T {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}

func rotateL[T mat.DType](a []T, i int) []T {
	x, b := a[:i], a[i:]
	return append(b, x...)
}
