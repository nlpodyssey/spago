// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/saientist/spago/pkg/mat"

// Identity function.
// y = x
type Identity struct {
	x Operand
}

func NewIdentity(x Operand) *Identity {
	return &Identity{x: x}
}

// Forward computes the output of the function.
func (r *Identity) Forward() mat.Matrix {
	return r.x.Value()
}

func (r *Identity) Backward(gy mat.Matrix) {
	r.x.PropagateGrad(gy)
}
