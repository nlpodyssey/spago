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
	// Name returns a human-readable label to identify or describe the Node.
	// It's optional and can be an empty string.
	//
	// This method is intended for introspection, debugging, and testing.
	// Identifying a Node solely upon its name is highly discouraged.
	Name() string
}

// ToNodes casts a slice []N into a slice []ag.Node.
func ToNodes[N Node](xs []N) []Node {
	ns := make([]Node, len(xs))
	for i, v := range xs {
		ns[i] = v
	}
	return ns
}

// CopyValue returns a copy of Node.Value.
// If Node.Value is nil, CopyValue returns nil as well.
//
// It is important to remember that Node.Value is a weak value, as the matrix
// derived from graph's operations can be freed (see ReleaseGraph).
func CopyValue(node Node) mat.Matrix {
	if node.Value() == nil {
		return nil
	}
	return node.Value().Clone()
}

// CopyValues calls CopyValue for each node of the slice.
func CopyValues(nodes []Node) []mat.Matrix {
	values := make([]mat.Matrix, len(nodes))
	for i, n := range nodes {
		values[i] = CopyValue(n)
	}
	return values
}

// CopyGrad returns a copy of Node.Grad.
// If Node.Grad is nil, CopyGrad returns nil as well.
//
// It is important to remember that Node.Grad is a weak value, as the matrix
// derived from graph's operations can be freed (see Node.ZeroGrad).
func CopyGrad(node Node) mat.Matrix {
	if node.Grad() == nil {
		return nil
	}
	return node.Grad().Clone()
}
