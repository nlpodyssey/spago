// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Concat is an operator to perform vector concatenation.
type Concat[O mat.Tensor] struct {
	xs    []O
	ySize int
}

// NewConcat returns a new Concat Function.
func NewConcat[O mat.Tensor](xs []O) *Concat[O] {
	return &Concat[O]{
		xs:    xs,
		ySize: 0, // assigned during the Forward()
	}
}

// Operands returns the list of operands.
func (r *Concat[O]) Operands() []O {
	return r.xs
}

// Forward computes the output of the function.
func (r *Concat[O]) Forward() (mat.Tensor, error) {
	if len(r.xs) == 0 {
		return nil, fmt.Errorf("fn: no vectors to concatenate")
	}
	r.ySize = 0 // reset output size
	ms := make([]mat.Matrix, len(r.xs))
	for i, x := range r.xs {
		value := x.Value().(mat.Matrix)
		ms[i] = value
		r.ySize += value.Size()
	}
	return ms[0].NewConcatV(ms...), nil
}

// Backward computes the backward pass.
func (r *Concat[O]) Backward(gy mat.Tensor) error {
	if r.ySize != gy.Size() {
		return fmt.Errorf("fn: vectors with not compatible size: %d != %d", r.ySize, gy.Size())
	}
	sizes := make([]int, len(r.xs))
	for i, x := range r.xs {
		sizes[i] = x.Value().Size()
	}
	xs := r.xs
	for i, gx := range gy.(mat.Matrix).SplitV(sizes...) {
		if xs[i].RequiresGrad() {
			xs[i].AccGrad(gx)
		}
	}
	return nil
}
