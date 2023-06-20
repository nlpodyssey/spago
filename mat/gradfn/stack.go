// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Stack is a Function which stacks together all given operand matrices,
// producing a single bigger matrix as result.
type Stack[O mat.Tensor] struct {
	xs []O
}

// NewStack returns a new Stack Function.
func NewStack[O mat.Tensor](xs []O) *Stack[O] {
	return &Stack[O]{xs: xs}
}

// Operands returns the list of operands.
func (r *Stack[O]) Operands() []O {
	return r.xs
}

// Forward computes the output of the function.
func (r *Stack[O]) Forward() (mat.Tensor, error) {
	if len(r.xs) == 0 {
		return nil, fmt.Errorf("fn: Stack has no operands")
	}
	vs := make([]mat.Matrix, len(r.xs))
	for i, x := range r.xs {
		vs[i] = x.Value().(mat.Matrix)
	}
	return vs[0].NewStack(vs...), nil
}

// Backward computes the backward pass.
func (r *Stack[O]) Backward(gy mat.Tensor) error {
	if gy.Shape()[0] != len(r.xs) {
		return fmt.Errorf("fn: matrices with not compatible size")
	}

	for i, x := range r.xs {
		if !x.RequiresGrad() {
			continue
		}
		gyRow := gy.(mat.Matrix).ExtractRow(i).ReshapeInPlace(x.Value().Shape()...)
		x.AccGrad(gyRow)
	}
	return nil
}
