// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Sigmoid is an operator to perform element-wise sigmoid function.
type Sigmoid[O mat.Tensor] struct {
	x O
}

// NewSigmoid returns a new Log Function.
func NewSigmoid[O mat.Tensor](x O) *Sigmoid[O] {
	return &Sigmoid[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (l *Sigmoid[O]) Operands() []mat.Tensor {
	return []mat.Tensor{l.x}
}

// Forward computes the output of the function.
func (l *Sigmoid[O]) Forward() (mat.Tensor, error) {
	// TODO: cache the sigmoid value in the forward pass for the backward pass?
	return l.x.Value().(mat.Matrix).Sigmoid(), nil
}

// Backward computes the backward pass.
func (l *Sigmoid[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(l.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if l.x.RequiresGrad() {
		gx := l.x.Value().(mat.Matrix).Sigmoid().Apply(func(_, _ int, v float64) float64 {
			return v * (1.0 - v) // derivative of the sigmoid function
		})
		gx.ProdInPlace(gy.(mat.Matrix))
		l.x.AccGrad(gx)
	}
	return nil
}
