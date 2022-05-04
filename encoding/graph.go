// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encoding

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// Graph provides different views over a set of "ag.Node"s being part
// of one computational graph (or even more disjointed graphs, depending on how
// it is initialized).
//
// The main purpose of this structure is to facilitate the visit of nodes and
// their connections from custom code, or go templates, for encoding and
// representing a graph into specific formats.
//
// It's often desirable to associate a unique identifier to each node. Since
// node don't have any special identifier, a simple numeric index/ID (int) is
// associated to each node during the Graph creation (see NewGraph).
// These numbers have no special meaning, and they are just incrementally
// assigned while each node is visited for the first time.
type Graph[T mat.DType] struct {
	// NodesList is a slice containing the unique nodes being part of a graph
	// (or graphs) discovered starting from the nodes passed to NewGraph.
	// Each slice index corresponds to the artificial ID of each node.
	NodesList []ag.Node[T]
	// NodesMap acts as an inverted index from NodesList. It associates each
	// unique Node to its artificial ID, that's also its index from NodesList.
	NodesMap map[ag.Node[T]]int
	// NodesByTimeStep maps each unique time step to the list of IDs of the
	// nodes associated to it.
	NodesByTimeStep map[int][]int
	// The Edges of the graph (or graphs), built by following the
	// relationships between operator-Nodes and their operands. To each
	// unique Node ID is associated a list of IDs of operator-Node IDs being
	// operands of the former.
	Edges map[int][]int
}

// NewGraph builds a new Graph starting from one or more given nodes
// (usually, the final output nodes of a computation).
func NewGraph[T mat.DType](nodes ...ag.Node[T]) *Graph[T] {
	g := &Graph[T]{
		NodesMap:        make(map[ag.Node[T]]int, 0),
		NodesList:       make([]ag.Node[T], 0),
		NodesByTimeStep: make(map[int][]int, 0),
		Edges:           make(map[int][]int, 0),
	}
	for _, n := range nodes {
		g.init(n)
	}
	return g
}

func (g *Graph[T]) init(n ag.Node[T]) {
	if _, exists := g.NodesMap[n]; exists {
		return
	}

	index := len(g.NodesList)
	g.NodesMap[n] = index
	g.NodesList = append(g.NodesList, n)

	ts := ag.TimeStep(n)
	g.NodesByTimeStep[ts] = append(g.NodesByTimeStep[ts], index)

	op, isOp := n.(*ag.Operator[T])
	if !isOp {
		return
	}

	operands := op.Operands()
	visitedOperands := make(map[int]struct{}, len(operands))
	for _, operand := range operands {
		g.init(operand)

		operandIndex := g.NodesMap[operand]
		if _, ok := visitedOperands[operandIndex]; ok {
			continue
		}
		g.Edges[operandIndex] = append(g.Edges[operandIndex], index)
		visitedOperands[operandIndex] = struct{}{}
	}
}
