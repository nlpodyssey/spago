// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Sigmoid is an operator to perform element-wise sigmoid function.
type Sigmoid[O DualValue] struct {
	x O
}

// NewSigmoid returns a new Log Function.
func NewSigmoid[O DualValue](x O) *Sigmoid[O] {
	return &Sigmoid[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (l *Sigmoid[O]) Operands() []O {
	return []O{l.x}
}

// Forward computes the output of the function.
func (l *Sigmoid[O]) Forward() (mat.Matrix, error) {
	// TODO: cache the sigmoid value in the forward pass for the backward pass?
	return l.x.Value().Sigmoid(), nil
}

// Backward computes the backward pass.
func (l *Sigmoid[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(l.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if l.x.RequiresGrad() {
		gx := l.x.Value().Sigmoid().Apply(func(_, _ int, v float64) float64 {
			return v * (1.0 - v) // derivative of the sigmoid function
		})
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		l.x.AccGrad(gx)
	}
	return nil
}
