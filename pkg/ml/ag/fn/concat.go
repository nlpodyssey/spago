// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &Concat{}

// Concat is an operator to perform vector concatenation.
type Concat struct {
	xs    []Operand
	ySize int
}

// NewConcat returns a new Concat Function.
func NewConcat(xs []Operand) *Concat {
	return &Concat{
		xs:    xs,
		ySize: 0, // assigned during the Forward()
	}
}

// Forward computes the output of the function.
func (r *Concat) Forward() mat.Matrix {
	r.ySize = 0 // reset output size
	ms := make([]mat.Matrix, len(r.xs))
	for i, x := range r.xs {
		value := x.Value()
		ms[i] = value
		r.ySize += value.Size()
	}
	return mat.ConcatV(ms...)
}

// Backward computes the backward pass.
func (r *Concat) Backward(gy mat.Matrix) {
	if r.ySize != gy.Size() {
		panic("fn: vectors with not compatible size")
	}
	sizes := make([]int, len(r.xs))
	for i, x := range r.xs {
		sizes[i] = x.Value().Size()
	}
	xs := r.xs
	for i, gx := range gy.SplitV(sizes...) {
		if xs[i].RequiresGrad() {
			xs[i].PropagateGrad(gx)
		}
		mat.ReleaseMatrix(gx)
	}
}
