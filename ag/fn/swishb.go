// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// SwishB function: f(x) = x * sigmoid.
//
// Reference: "Searching for Activation Functions" by Ramachandran et al, 2017.
// (https://arxiv.org/pdf/1710.05941.pdf)
type SwishB[O Operand] struct {
	x        O
	beta     O // scalar
	operands []O
}

// NewSwishB returns a new SwishB Function.
func NewSwishB[O Operand](x O, beta O) *SwishB[O] {
	return &SwishB[O]{
		x:        x,
		beta:     beta,
		operands: []O{x, beta},
	}
}

// Operands returns the list of operands.
func (r *SwishB[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *SwishB[O]) Forward() mat.Matrix {
	y := r.x.Value().ApplyWithAlpha(swishB, r.beta.Value().Scalar().Float64())
	return y
}

// Backward computes the backward pass.
func (r *SwishB[O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(swishBDeriv, r.beta.Value().Scalar().Float64())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
	if r.beta.RequiresGrad() {
		gb := r.beta.Value().ZerosLike()
		defer mat.ReleaseMatrix(gb)
		// FIXME: avoid casting to specific type
		for i, x := range r.x.Value().Data().Float64() {
			deriv := swishBBetaDeriv(x, r.beta.Value().Scalar().Float64())
			gyi := gy.ScalarAtVec(i).Float64()
			gb.AddScalarInPlace(deriv * gyi)
		}
		r.beta.AccGrad(gb)
	}
}
