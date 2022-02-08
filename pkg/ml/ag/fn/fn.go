// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

// Operand is implemented by any value that implements automatic differentiation features.
type Operand interface {
	// Value returns the value of the operand.
	Value() mat.Matrix[mat.Float]
	// PropagateGrad propagates the gradients gx to the operands.
	PropagateGrad(gx mat.Matrix[mat.Float])
	// RequiresGrad returns true if the operand requires gradients.
	RequiresGrad() bool
}

// Function represents a function with automatic differentiation features.
type Function interface {
	// Forward computes the output of the function.
	Forward() mat.Matrix[mat.Float]
	// Backward computes the backward pass.
	Backward(gy mat.Matrix[mat.Float])
}
