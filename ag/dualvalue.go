// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/mat"

type DualValue interface {
	Node[mat.Matrix]
}

// DualValue defines the interface for a node in the computational graph.
type Node[T any] interface {
	// Value returns the value of the node.
	// In case of a leaf node, it returns the value of the underlying matrix.
	// In case of a non-leaf node, it returns the value of the operation performed during the forward pass.
	Value() T
	// Grad returns the gradients accumulated during the backward pass.
	// A matrix full of zeros and the nil value are considered equivalent.
	Grad() T
	// HasGrad reports whether there are accumulated gradients.
	HasGrad() bool
	// RequiresGrad reports whether the node requires gradients.
	RequiresGrad() bool
	// AccGrad accumulates the gradients into the node.
	AccGrad(gx T)
	// ZeroGrad zeroes the gradients, setting the value of Grad to nil.
	ZeroGrad()
}