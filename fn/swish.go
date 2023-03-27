// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat"
)

// Swish is an operator to perform element-wise swish function: y = x * sigmoid(x).
type Swish[O DualValue] struct {
	x O
}

// NewSwish returns a new Swish Function.
func NewSwish[O DualValue](x O) *Swish[O] {
	return &Swish[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (l *Swish[O]) Operands() []O {
	return []O{l.x}
}

// Forward computes the output of the function.
func (l *Swish[O]) Forward() (mat.Matrix, error) {
	x := l.x.Value()
	s := x.Sigmoid()
	return s.ProdInPlace(x), nil
}

// Backward computes the backward pass.
func (l *Swish[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(l.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if l.x.RequiresGrad() {
		gx := l.x.Value().Apply(swishDeriv)
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		l.x.AccGrad(gx)
	}
	return nil
}

func swishDeriv(_, _ int, v float64) float64 {
	exp := math.Exp(v)
	expPlusOne := exp + 1
	return exp * (expPlusOne + v) / (expPlusOne * expPlusOne)
}
