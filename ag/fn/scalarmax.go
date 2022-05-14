// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// ScalarMax is an operator to perform reduce-max function on a list of scalars.
// It gets the maximum element of the Operand x
type ScalarMax[O Operand] struct {
	xs     []O
	argmax int
}

// NewScalarMax returns a new ScalarMax Function.
func NewScalarMax[O Operand](xs []O) *ScalarMax[O] {
	return &ScalarMax[O]{xs: xs}
}

// Operands returns the list of operands.
func (r *ScalarMax[O]) Operands() []O {
	return r.xs
}

// Forward computes the output of this function.
func (r *ScalarMax[O]) Forward() mat.Matrix {
	if len(r.xs) == 0 {
		panic("fn: ScalarMax has no operands")
	}
	var max float64
	var argmax int
	for i, x := range r.xs {
		// FIXME: avoid casting to specific type
		val := x.Value().Scalar().Float64()
		if val > max {
			max = val
			argmax = i
		}
	}
	r.argmax = argmax
	return r.xs[argmax].Value().Clone()
}

// Backward computes the backward pass.
func (r *ScalarMax[O]) Backward(gy mat.Matrix) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	target := r.xs[r.argmax]
	if target.RequiresGrad() {
		target.AccGrad(gy)
	}
}
