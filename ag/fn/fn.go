// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Operand is implemented by any value that implements automatic differentiation features.
type Operand[T mat.DType] interface {
	// Value returns the value of the operand.
	Value() mat.Matrix
	// AccGrad accumulate the gradients gx to the operands.
	AccGrad(gx mat.Matrix)
	// RequiresGrad returns true if the operand requires gradients.
	RequiresGrad() bool
}

// Function represents a function with automatic differentiation features.
type Function[T mat.DType, O Operand[T]] interface {
	// Forward computes the output of the function.
	Forward() mat.Matrix
	// Backward computes the backward pass.
	Backward(gy mat.Matrix)
	// Operands returns the list of operands.
	Operands() []O
}
