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
