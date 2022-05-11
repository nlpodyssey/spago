// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Pow is an operator to perform element-wise pow function.
type Pow[T mat.DType, O Operand[T]] struct {
	x        O
	power    float64
	operands []O
}

// NewPow returns a new Pow Function.
func NewPow[T mat.DType, O Operand[T]](x O, power float64) *Pow[T, O] {
	return &Pow[T, O]{
		x:        x,
		power:    power,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *Pow[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Pow[T, O]) Forward() mat.Matrix[T] {
	return r.x.Value().Pow(r.power)
}

// Backward computes the backward pass.
func (r *Pow[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().Pow(r.power - 1)
		defer mat.ReleaseMatrix(gx)
		gx.ProdScalarInPlace(T(r.power)).ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
