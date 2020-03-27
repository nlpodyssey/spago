// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

// Swish(x) = ​x * sigmoid
// Reference: "Searching for Activation Functions" by Ramachandran et al, 2017.
// (https://arxiv.org/pdf/1710.05941.pdf)
type Swish struct {
	x    Operand
	beta Operand // scalar
}

func NewSwish(x, beta Operand) *Swish {
	return &Swish{x: x, beta: beta}
}

// Forward computes the output of the function.
func (r *Swish) Forward() mat.Matrix {
	y := r.x.Value().ZerosLike()
	y.ApplyWithAlpha(swish, r.x.Value(), r.beta.Value().Scalar())
	return y
}

func (r *Swish) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		gx.ApplyWithAlpha(swishDeriv, r.x.Value(), r.beta.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
	if r.beta.RequiresGrad() {
		gb := r.beta.Value().ZerosLike()
		for i, x := range r.x.Value().Data() {
			gb.AddScalarInPlace(swishBetaDeriv(x, r.beta.Value().Scalar()) * gy.Data()[i])
		}
		r.beta.PropagateGrad(gb)
	}
}
