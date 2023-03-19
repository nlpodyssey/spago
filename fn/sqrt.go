// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Sqrt is an operator to perform element-wise square root function.
type Sqrt[O DualValue] struct {
	x O
}

// NewSqrt returns a new Sqrt Function.
func NewSqrt[O DualValue](x O) *Sqrt[O] {
	return &Sqrt[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Sqrt[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *Sqrt[O]) Forward() mat.Matrix {
	return r.x.Value().Sqrt()
}

// Backward computes the backward pass.
func (r *Sqrt[O]) Backward(gy mat.Matrix) {
	if !mat.SameDims(r.x.Value(), gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().Pow(-.5)
		defer mat.ReleaseMatrix(gx)
		gx.ProdScalarInPlace(.5)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
