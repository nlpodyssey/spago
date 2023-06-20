// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Log is an operator to perform element-wise natural logarithm function.
type Log[O mat.Tensor] struct {
	x O
}

// NewLog returns a new Log Function.
func NewLog[O mat.Tensor](x O) *Log[O] {
	return &Log[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (l *Log[O]) Operands() []mat.Tensor {
	return []mat.Tensor{l.x}
}

// Forward computes the output of the function.
func (l *Log[O]) Forward() (mat.Tensor, error) {
	return l.x.Value().(mat.Matrix).Log(), nil
}

// Backward computes the backward pass.
func (l *Log[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(l.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if l.x.RequiresGrad() {
		gx := l.x.Value().(mat.Matrix).Apply(safeLogDeriv)
		gx.ProdInPlace(gy.(mat.Matrix))
		l.x.AccGrad(gx)
	}
	return nil
}

func safeLogDeriv(_, _ int, v float64) float64 {
	if v > 0.0 {
		return 1.0 / v
	}
	if v == 0.0 {
		return 1.0 / 1.0e-08
	}
	panic("ag: invalid log for negative values")
}
