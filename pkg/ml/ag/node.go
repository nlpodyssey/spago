// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/pkg/ml/ag/fn"

// Node is implemented by any value that can represent a node of a Graph.
type Node interface {
	GradValue
	// Graph returns the graph this node belongs to.
	Graph() *Graph
	// ID returns the ID of the node in the graph.
	ID() int
	// TimeStep returns the time-step associated to this node.
	TimeStep() int
}

// Operands cast a slice of nodes into a slice of operands.
func Operands(xs []Node) []fn.Operand {
	var out = make([]fn.Operand, len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}
