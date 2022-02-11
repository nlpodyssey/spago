// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

// variable used in the tests
type variable[T mat.DType] struct {
	value        mat.Matrix[T]
	grad         mat.Matrix[T]
	requiresGrad bool
}

func (v *variable[T]) Value() mat.Matrix[T]           { return v.value }
func (v *variable[T]) PropagateGrad(gx mat.Matrix[T]) { v.grad = gx.Clone() }
func (v *variable[_]) RequiresGrad() bool             { return v.requiresGrad }
