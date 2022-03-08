// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

// Node is implemented by any value that can represent a node of a Graph.
type Node[T mat.DType] interface {
	GradValue[T]
	// Graph returns the graph this node belongs to.
	Graph() *Graph[T]
	// ID returns the ID of the node in the graph.
	ID() int
	// TimeStep returns the time-step associated to this node.
	TimeStep() int
}

type _node[T mat.DType] interface {
	Node[T]
	setID(int)
}

// ToNodes cast a slice of N[T] into a slice of ag.Node.
func ToNodes[T mat.DType, N Node[T]](xs []N) []Node[T] {
	ns := make([]Node[T], len(xs))
	for i, v := range xs {
		ns[i] = v
	}
	return ns
}

// ToOperands cast a slice of nodes into a slice of operands.
func ToOperands[T mat.DType](xs []Node[T]) []fn.Operand[T] {
	var out = make([]fn.Operand[T], len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}

// CopyValue returns a copy of the value of a Node. If the value is nil, CopyValue returns nil as well.
// The returned value is a copy, so it is safe to use even after the graph has been cleared calling Graph.Clear().
// It is important to remember that the Value() property of a Node is a weak access, as the matrix derived from
// graph's operations can be freed.
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

// CopyGrad returns a copy of the gradients of a Node. If the gradients are nil, CopyGrad returns nil as well.
// The returned value is a copy, so it is safe to use even after the graph has been cleared calling Graph.Clear().
// It is important to remember that the Grad() property of a Node is a weak access, as the matrix derived from
// graph's operations can be freed.
func CopyGrad[T mat.DType](node Node[T]) mat.Matrix[T] {
	if node.Grad() == nil {
		return nil
	}
	return node.Grad().Clone()
}

// ReplaceValue replaces the current value of a variable Node with the given value.
// It panics if node is not a variable.
func ReplaceValue[T mat.DType](node Node[T], value mat.Matrix[T]) {
	if node, ok := node.(*Variable[T]); !ok {
		panic("ag: invalid node. Only variables are allowed to change their value.")
	} else {
		node.value = value
	}
}
