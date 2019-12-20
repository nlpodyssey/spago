// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
)

type Pow struct {
	x     Operand
	power float64
}

func NewPow(x Operand, power float64) *Pow {
	return &Pow{x: x, power: power}
}

// Forward computes the output of the function.
func (r *Pow) Forward() mat.Matrix {
	return r.x.Value().Pow(r.power)
}

func (r *Pow) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		r.x.PropagateGrad(r.x.Value().Pow(r.power - 1).ProdScalar(r.power).Prod(gy))
	}
}
