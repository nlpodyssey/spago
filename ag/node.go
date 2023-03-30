// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/mat"

// Node is implemented by any value that can represent a node of a graph.
type Node interface {
	// Value returns the value of the node.
	// If the node is a variable it returns its value, otherwise returns the
	// cached result of the forward pass.
	Value() mat.Matrix
	// Grad returns the gradients accumulated during the backward pass.
	// A matrix full of zeros and the nil value are considered equivalent.
	Grad() mat.Matrix
	// HasGrad reports whether there are accumulated gradients.
	HasGrad() bool
	// RequiresGrad reports whether the node requires gradients.
	RequiresGrad() bool
	// AccGrad accumulates the gradients into the node.
	AccGrad(gx mat.Matrix)
	// ZeroGrad zeroes the gradients, setting the value of Grad to nil.
	ZeroGrad()
}
