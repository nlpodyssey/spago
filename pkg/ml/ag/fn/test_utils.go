// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

// variable used in the tests
type variable struct {
	value        mat.Matrix[mat.Float]
	grad         mat.Matrix[mat.Float]
	requiresGrad bool
}

func (v *variable) Value() mat.Matrix[mat.Float]           { return v.value }
func (v *variable) PropagateGrad(gx mat.Matrix[mat.Float]) { v.grad = gx.Clone() }
func (v *variable) RequiresGrad() bool                     { return v.requiresGrad }
