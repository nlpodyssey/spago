// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// The Graph a.k.a. expression graph or computational graph is the centerpiece of the spaGO machine learning framework.
// It takes the form of a directed graph with no directed cycles (DAG).
type Graph[T mat.DType] struct {
	// to avoid data race during concurrent computations (mu2 is used in Constant())
	mu, mu2 sync.Mutex
	// the time-step is useful to perform truncated back propagation (default 0)
	curTimeStep int
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	// The nodes are inserted one at a time in order of creation.
	nodes []Node[T]
	// constants maps scalar values that that doesn't require gradients to a Node. It is used in the Constant() method.
	constants map[T]Node[T]
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.WithRand.
func NewGraph[T mat.DType]() *Graph[T] {
	return &Graph[T]{
		curTimeStep: 0,
		nodes:       nil,
		constants:   map[T]Node[T]{},
	}
}

// Clear cleans the graph and should be called after the graph has been used.
// It releases the matrices underlying the operator nodes so the node operators become weak references.
// It is therefore recommended to make always a copy of the results of node.Value().
// You can use the convenient ag.CopyValue(node) and ag.CopyGrad(node)
func (g *Graph[T]) Clear() {
	WaitForOngoingComputations() // this awaits computation of other graphs as well
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.curTimeStep = 0
	g.nodes = nil
}

// TimeStep is an integer value associated with the graph, which can be useful
// to perform truncated back propagation. This value is 0 for a new Graph, and
// can be incremented calling IncTimeStep.
func (g *Graph[_]) TimeStep() int {
	return g.curTimeStep
}

// IncTimeStep increments the value of the graph's TimeStep by one.
func (g *Graph[_]) IncTimeStep() {
	g.curTimeStep++
}

// insert append the node into the graph's nodes.
func (g *Graph[T]) insert(n Node[T]) Node[T] {
	g.mu.Lock()
	g.nodes = append(g.nodes, n)
	g.mu.Unlock()
	return n
}
