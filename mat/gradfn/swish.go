// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat"
)

// Swish is an operator to perform element-wise swish function: y = x * sigmoid(x).
type Swish[O mat.Tensor] struct {
	x O
}

// NewSwish returns a new Swish Function.
func NewSwish[O mat.Tensor](x O) *Swish[O] {
	return &Swish[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (l *Swish[O]) Operands() []mat.Tensor {
	return []mat.Tensor{l.x}
}

// Forward computes the output of the function.
func (l *Swish[O]) Forward() (mat.Tensor, error) {
	x := l.x.Value().(mat.Matrix)
	s := x.Sigmoid()
	return s.ProdInPlace(x), nil
}

// Backward computes the backward pass.
func (l *Swish[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(l.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if l.x.RequiresGrad() {
		gx := l.x.Value().(mat.Matrix).Apply(swishDeriv)
		gx.ProdInPlace(gy.(mat.Matrix))
		l.x.AccGrad(gx)
	}
	return nil
}

func swishDeriv(_, _ int, v float64) float64 {
	exp := math.Exp(v)
	expPlusOne := exp + 1
	return exp * (expPlusOne + v) / (expPlusOne * expPlusOne)
}
