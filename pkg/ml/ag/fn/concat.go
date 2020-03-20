// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/saientist/spago/pkg/mat"
)

type Concat struct {
	xs []Operand
}

func NewConcat(xs []Operand) *Concat {
	return &Concat{xs: xs}
}

// Forward computes the output of the function.
func (r *Concat) Forward() mat.Matrix {
	ms := make([]mat.Matrix, len(r.xs))
	for i, x := range r.xs {
		ms[i] = x.Value()
	}
	return mat.ConcatV(ms...)
}

func (r *Concat) Backward(gy mat.Matrix) {
	sizes := make([]int, len(r.xs))
	for i, x := range r.xs {
		sizes[i] = x.Value().Size()
	}
	for i, gx := range gy.(*mat.Dense).SplitV(sizes...) {
		if r.xs[i].RequiresGrad() {
			r.xs[i].PropagateGrad(gx)
		}
	}
}
