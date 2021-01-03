// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import mat "github.com/nlpodyssey/spago/pkg/mat32"

// variable used in the tests
type variable struct {
	value        mat.Matrix
	grad         mat.Matrix
	requiresGrad bool
}

func (v *variable) Value() mat.Matrix           { return v.value }
func (v *variable) PropagateGrad(gx mat.Matrix) { v.grad = gx.Clone() }
func (v *variable) RequiresGrad() bool          { return v.requiresGrad }
