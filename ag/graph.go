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
	// maxID is the id of the last inserted node (corresponds of len(nodes)-1)
	maxID int
	// the time-step is useful to perform truncated back propagation (default 0)
	curTimeStep int
	// timeStepBoundaries holds the first node associated to each time-step
	timeStepBoundaries []int
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	// The nodes are inserted one at a time in order of creation.
	nodes []Node[T]
	// constants maps scalar values that that doesn't require gradients to a Node. It is used in the Constant() method.
	constants map[T]Node[T]
	// fWG waits for the forward goroutines to finish.
	fWG *sync.WaitGroup
	// bWG waits for the backward goroutines to finish.
	bWG *sync.WaitGroup
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.WithRand.
func NewGraph[T mat.DType]() *Graph[T] {
	return &Graph[T]{
		maxID:              -1,
		curTimeStep:        0,
		timeStepBoundaries: []int{0},
		nodes:              nil,
		constants:          map[T]Node[T]{},
		fWG:                &sync.WaitGroup{},
		bWG:                &sync.WaitGroup{},
	}
}

// Clear cleans the graph and should be called after the graph has been used.
// It releases the matrices underlying the operator nodes so the node operators become weak references.
// It is therefore recommended to make always a copy of the results of node.Value().
// You can use the convenient ag.CopyValue(node) and ag.CopyGrad(node)
func (g *Graph[T]) Clear() {
	g.fWG.Wait()
	g.bWG.Wait()
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.maxID = -1
	g.curTimeStep = 0
	g.timeStepBoundaries = []int{0}
	g.releaseMemory()
	g.nodes = nil
}

// Nodes returns the nodes of the graph.
func (g *Graph[T]) Nodes() []Node[T] {
	return g.nodes
}

// ZeroGrad sets the gradients of all nodes to zero.
func (g *Graph[_]) ZeroGrad() {
	for _, node := range g.nodes {
		node.ZeroGrad()
	}
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
	g.timeStepBoundaries = append(g.timeStepBoundaries, g.maxID+1)
}

// releaseMemory clears the values and the gradients of operator nodes.
// Since the values and the gradients within the nodes are handled through a pool of dense matrices,
// releasing them allows the memory to be reused without being reallocated, improving performance.
func (g *Graph[T]) releaseMemory() {
	g.fWG.Wait()
	g.bWG.Wait()
	for _, node := range g.nodes {
		op, ok := node.(*Operator[T])
		if !ok {
			continue
		}
		op.releaseValue()
		op.ZeroGrad()
		*op = Operator[T]{} // free operator
		getOperatorPool[T]().Put(op)
		// TODO: release constants?
	}
}

// insert append the node into the graph's nodes and assign it an id.
func (g *Graph[T]) insert(n nodeInternal[T]) Node[T] {
	g.mu.Lock()
	g.maxID++
	n.setID(g.maxID)
	g.nodes = append(g.nodes, n)
	g.mu.Unlock()
	return n
}

func (g *Graph[T]) nodeBoundaries(fromTimeStep, toTimeStep int) (startNodeIndex, endNodeIndex int) {
	startNodeIndex = g.timeStepBoundaries[fromTimeStep] // inclusive
	endNodeIndex = len(g.nodes)                         // exclusive

	if toTimeStep != -1 && toTimeStep != g.TimeStep() {
		endNodeIndex = g.timeStepBoundaries[toTimeStep+1]
	}
	return
}
