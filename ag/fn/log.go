// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Log is an operator to perform element-wise natural logarithm function.
type Log[O Operand] struct {
	x        O
	operands []O
}

// NewLog returns a new Log Function.
func NewLog[O Operand](x O) *Log[O] {
	return &Log[O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (l *Log[O]) Operands() []O {
	return l.operands
}

// Forward computes the output of the function.
func (l *Log[O]) Forward() mat.Matrix {
	return l.x.Value().Log()
}

// Backward computes the backward pass.
func (l *Log[O]) Backward(gy mat.Matrix) {
	if !mat.SameDims(l.x.Value(), gy) {
		panic("fn: matrices have incompatible dimensions")
	}
	if l.x.RequiresGrad() {
		gx := l.x.Value().Apply(safeLogDeriv)
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		l.x.AccGrad(gx)
	}
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
