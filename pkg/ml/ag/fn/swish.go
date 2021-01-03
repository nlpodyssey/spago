// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &Swish{}

// Swish function: f(x) = x * sigmoid.
//
// Reference: "Searching for Activation Functions" by Ramachandran et al, 2017.
// (https://arxiv.org/pdf/1710.05941.pdf)
type Swish struct {
	x    Operand
	beta Operand // scalar
}

// NewSwish returns a new Swish Function.
func NewSwish(x, beta Operand) *Swish {
	return &Swish{x: x, beta: beta}
}

// Forward computes the output of the function.
func (r *Swish) Forward() mat.Matrix {
	y := mat.GetDenseWorkspace(r.x.Value().Dims())
	y.ApplyWithAlpha(swish, r.x.Value(), r.beta.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *Swish) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDenseWorkspace(r.x.Value().Dims())
		gx.ApplyWithAlpha(swishDeriv, r.x.Value(), r.beta.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
	if r.beta.RequiresGrad() {
		gb := mat.GetDenseWorkspace(r.beta.Value().Dims())
		defer mat.ReleaseDense(gb)
		for i, x := range r.x.Value().Data() {
			gb.AddScalarInPlace(swishBetaDeriv(x, r.beta.Value().Scalar()) * gy.Data()[i])
		}
		r.beta.PropagateGrad(gb)
	}
}
