// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/saientist/spago/pkg/mat"
)

// Single-input, reduce mean function.
type ReduceMean struct {
	x Operand
}

func NewReduceMean(x Operand) *ReduceMean {
	return &ReduceMean{x: x}
}

// Forward computes the output of this node.
func (r *ReduceMean) Forward() mat.Matrix {
	return mat.NewScalar(r.x.Value().Sum() / float64(r.x.Value().Size()))
}

func (r *ReduceMean) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		r.x.PropagateGrad(mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar()/float64(r.x.Value().Size())))
	}
}
