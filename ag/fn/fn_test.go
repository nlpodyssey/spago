// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// variable is a simple implementation satisfying the Operand interface.
type variable[T mat.DType] struct {
	value        mat.Matrix
	grad         mat.Matrix
	requiresGrad bool
}

func (v *variable[T]) Value() mat.Matrix {
	return v.value
}

func (v *variable[T]) AccGrad(gx mat.Matrix) {
	if v.grad == nil {
		v.grad = gx.Clone()
		return
	}
	v.grad.AddInPlace(gx)
}

func (v *variable[_]) RequiresGrad() bool {
	return v.requiresGrad
}
