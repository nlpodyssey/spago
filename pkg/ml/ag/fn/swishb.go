// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &SwishB[float32]{}

// SwishB function: f(x) = x * sigmoid.
//
// Reference: "Searching for Activation Functions" by Ramachandran et al, 2017.
// (https://arxiv.org/pdf/1710.05941.pdf)
type SwishB[T mat.DType] struct {
	x    Operand[T]
	beta Operand[T] // scalar
}

// NewSwishB returns a new SwishB Function.
func NewSwishB[T mat.DType](x, beta Operand[T]) *SwishB[T] {
	return &SwishB[T]{x: x, beta: beta}
}

// Forward computes the output of the function.
func (r *SwishB[T]) Forward() mat.Matrix[T] {
	y := mat.GetDensePool[T]().Get(r.x.Value().Dims())
	y.ApplyWithAlpha(swishB[T], r.x.Value(), r.beta.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SwishB[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[T]().Get(r.x.Value().Dims())
		gx.ApplyWithAlpha(swishBDeriv[T], r.x.Value(), r.beta.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
	if r.beta.RequiresGrad() {
		gb := mat.GetDensePool[T]().Get(r.beta.Value().Dims())
		defer mat.ReleaseDense(gb)
		for i, x := range r.x.Value().Data() {
			gb.AddScalarInPlace(swishBBetaDeriv(x, r.beta.Value().Scalar()) * gy.Data()[i])
		}
		r.beta.PropagateGrad(gb)
	}
}
