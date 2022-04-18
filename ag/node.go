// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
)

// Node is implemented by any value that can represent a node of a graph.
type Node[T mat.DType] interface {
	// Value returns the value of the node.
	// If the node is a variable it returns its value, otherwise returns the
	// cached result of the forward pass.
	Value() mat.Matrix[T]
	// Grad returns the gradients accumulated during the backward pass.
	// A matrix full of zeros and the nil value are considered equivalent.
	Grad() mat.Matrix[T]
	// HasGrad reports whether there are accumulated gradients.
	HasGrad() bool
	// RequiresGrad reports whether the node requires gradients.
	RequiresGrad() bool
	// AccGrad accumulates the gradients into the node.
	AccGrad(gx mat.Matrix[T])
	// ZeroGrad zeroes the gradients, setting the value of Grad to nil.
	ZeroGrad()
	// TimeStep returns the time-step associated to this node.
	TimeStep() int
	// IncTimeStep increments the value of the node's TimeStep by one.
	IncTimeStep()
}

// ToNodes casts a slice of N[T] into a slice of ag.Node.
func ToNodes[T mat.DType, N Node[T]](xs []N) []Node[T] {
	ns := make([]Node[T], len(xs))
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
func CopyValue[T mat.DType](node Node[T]) mat.Matrix[T] {
	if node.Value() == nil {
		return nil
	}
	return node.Value().Clone()
}

// CopyValues calls CopyValue for each node of the slice.
func CopyValues[T mat.DType](nodes []Node[T]) []mat.Matrix[T] {
	values := make([]mat.Matrix[T], len(nodes))
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
func CopyGrad[T mat.DType](node Node[T]) mat.Matrix[T] {
	if node.Grad() == nil {
		return nil
	}
	return node.Grad().Clone()
}
