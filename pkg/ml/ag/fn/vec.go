// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

type Vec struct {
	x Operand
}

func NewVec(x Operand) *Vec {
	return &Vec{x: x}
}

// Forward computes the output of the node.
func (r *Vec) Forward() mat.Matrix {
	return r.x.Value().Reshape(r.x.Value().Size(), 1)
}

func (r *Vec) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		r.x.PropagateGrad(gy.Reshape(r.x.Value().Dims()))
	}
}
