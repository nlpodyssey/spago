// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

type Transpose struct {
	x Operand
}

func NewTranspose(x Operand) *Transpose {
	return &Transpose{x: x}
}

// Forward computes the output of the node.
func (r *Transpose) Forward() mat.Matrix {
	return r.x.Value().T()
}

func (r *Transpose) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		r.x.PropagateGrad(gy.T())
	}
}
