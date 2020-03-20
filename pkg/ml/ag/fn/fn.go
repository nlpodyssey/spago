// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/saientist/spago/pkg/mat"

type Operand interface {
	// Value returns the value of the operand.
	Value() mat.Matrix
	// PropagateGrad propagates the gradients to the operands.
	PropagateGrad(gx mat.Matrix)
	// RequiresGrad returns true if the operand requires gradients.
	RequiresGrad() bool
}

type Function interface {
	// Forward computes the output of the function.
	Forward() mat.Matrix
	// Backward computes the backward pass.
	Backward(gy mat.Matrix)
}
