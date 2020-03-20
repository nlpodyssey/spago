// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/saientist/spago/pkg/mat"

// variable used in the tests
type variable struct {
	value        mat.Matrix
	grad         mat.Matrix
	requiresGrad bool
}

func (v *variable) Value() mat.Matrix           { return v.value }
func (v *variable) PropagateGrad(gx mat.Matrix) { v.grad = gx }
func (v *variable) RequiresGrad() bool          { return v.requiresGrad }
