// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/pkg/mat"

type GradValue interface {
	// Value returns the value of the node.
	// If the node is a variable it returns its value, otherwise returns the cached result of the forward pass.
	Value() mat.Matrix
	// ScalarValue() returns the scalar value of the node. It panics if the value is not a scalar.
	ScalarValue() float64
	// Grad returns the gradients accumulated during the backward pass.
	Grad() mat.Matrix
	// HasGrad returns true if there are accumulated gradients.
	HasGrad() bool
	// RequiresGrad returns true if the node requires gradients.
	RequiresGrad() bool
	// PropagateGrad propagates the gradients to the node.
	PropagateGrad(gx mat.Matrix)
	// ZeroGrad set the gradients to zeros.
	ZeroGrad()
}
