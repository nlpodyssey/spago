// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import "github.com/nlpodyssey/spago/mat"

// variable is a simple implementation satisfying the Operand interface.
type variable struct {
	value        mat.Matrix
	grad         mat.Matrix
	requiresGrad bool
}

// newDualValue creates a new variable with given value, and requiresGrad
// set to true.
func newDualValue(v mat.Matrix) *variable {
	return &variable{
		value:        v,
		requiresGrad: true,
		grad:         nil,
	}
}

func (v *variable) Value() mat.Matrix {
	return v.value
}

func (v *variable) AccGrad(gx mat.Matrix) {
	if v.grad == nil {
		v.grad = gx.Clone()
		return
	}
	v.grad.AddInPlace(gx)
}

func (v *variable) RequiresGrad() bool {
	return v.requiresGrad
}
