// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/mat"

// GradValue extends the fn.Operand interface providing more convenient methods
// to handle gradients in the context of automatic differentiation.
type GradValue[T mat.DType] interface {
	// Value returns the value of the node.
	// If the node is a variable it returns its value, otherwise returns the cached result of the forward pass.
	Value() mat.Matrix[T]
	// Grad returns the gradients accumulated during the backward pass.
	Grad() mat.Matrix[T]
	// HasGrad returns true if there are accumulated gradients.
	HasGrad() bool
	// RequiresGrad returns true if the node requires gradients.
	RequiresGrad() bool
	// AccGrad accumulate the gradients to the node.
	AccGrad(gx mat.Matrix[T])
	// ZeroGrad set the gradients to zeros.
	ZeroGrad()
}
