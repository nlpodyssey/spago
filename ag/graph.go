// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"sync"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
)

// The Graph a.k.a. expression graph or computational graph is the centerpiece of the spaGO machine learning framework.
// It takes the form of a directed graph with no directed cycles (DAG).
type Graph[T mat.DType] struct {
	// randGen is the generator of random numbers
	randGen *rand.LockedRand[T]
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
	// backwardInProgress regulates the claim and the propagation of gradients during the backpropagation.
	backwardInProgress bool
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.WithRand.
func NewGraph[T mat.DType](opts ...GraphOption[T]) *Graph[T] {
	g := &Graph[T]{
		maxID:              -1,
		curTimeStep:        0,
		timeStepBoundaries: []int{0},
		nodes:              nil,
		constants:          map[T]Node[T]{},
		fWG:                &sync.WaitGroup{},
		bWG:                &sync.WaitGroup{},
		backwardInProgress: false,
	}
	for _, opt := range opts {
		opt(g)
	}
	if g.randGen == nil {
		g.randGen = rand.NewLockedRand[T](1) // set default random generator
	}
	return g
}

// SetRand replace the graph's random number generator with the given one.
func (g *Graph[T]) SetRand(rand *rand.LockedRand[T]) {
	g.randGen = rand
}

// SetRandSeed replace the graph's random number generator with a new one with the given seed.
func (g *Graph[T]) SetRandSeed(seed uint64) {
	g.randGen = rand.NewLockedRand[T](seed)
}

// ZeroGrad sets the gradients of all nodes to zero.
func (g *Graph[_]) ZeroGrad() {
	for _, node := range g.nodes {
		node.ZeroGrad()
	}
}

// IncTimeStep increments the value of the graph's TimeStep by one.
func (g *Graph[_]) IncTimeStep() {
	g.curTimeStep++
	g.timeStepBoundaries = append(g.timeStepBoundaries, g.maxID+1)
}

// TimeStep is an integer value associated with the graph, which can be useful
// to perform truncated back propagation. This value is 0 for a new Graph, and
// can be incremented calling IncTimeStep.
func (g *Graph[_]) TimeStep() int {
	return g.curTimeStep
}

// Nodes returns the nodes of the graph.
func (g *Graph[T]) Nodes() []Node[T] {
	return g.nodes
}

// Clear cleans the graph. This is a destructive operation.
// It is not mandatory to call this method, but it is strongly recommended to do so when you finish using the graph.
// The cleaning of the graph improves the memory management and therefore the efficiency of execution.
// Clear releases the matrices underlying the nodes so to reduce the need of future new time-consuming allocations.
// It is important to stress that calling g.Clean(), the "value" and "grad" of the operators nodes are freed (set to nil).
// Whoever is using the Value() or Grad() properties of a node, does so at his own risk. It is therefore recommended to
// make always a copy of the return value of Value() or Grad().
// Alternatively, you can use the convenient graph's methods g.CopyValue(node) and g.CopyGrad(node).
func (g *Graph[T]) Clear() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.maxID = -1
	g.curTimeStep = 0
	g.timeStepBoundaries = []int{0}
	g.releaseMemory(false)
	g.nodes = nil
}

// ClearForReuse does the same thing as Clear(), with the difference that the graph structure (i.e.
// how nodes are connected to each other) is maintained.
// This allows you to efficiently use the graph as if it were "pre-computed" (see the Forward()
// method for this scenario).
func (g *Graph[_]) ClearForReuse() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.releaseMemory(true)
}

// NewVariable creates and returns a new node.
func (g *Graph[T]) NewVariable(value mat.Matrix[T], requiresGrad bool) Node[T] {
	n := &Variable[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
	}
	return g.insert(n)
}

// NewVariableWithName creates and returns a new node.
func (g *Graph[T]) NewVariableWithName(value mat.Matrix[T], requiresGrad bool, name string) Node[T] {
	n := &Variable[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		name:         name,
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
	}
	return g.insert(n)
}

// NewScalar creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph[T]) NewScalar(value T) Node[T] {
	return g.NewVariable(mat.NewScalar(value), false)
}

// NewScalarWithName creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph[T]) NewScalarWithName(value T, name string) Node[T] {
	return g.NewVariableWithName(mat.NewScalar(value), false, name)
}

// Constant returns a scalar Node that that doesn't require gradients.
// For the same value, a previously created Node is returned without creating a new one.
// Useful for example in the case of epsilon and number like 0.0 or 1.0.
func (g *Graph[T]) Constant(value T) Node[T] {
	g.mu2.Lock()
	defer g.mu2.Unlock()
	if node, ok := g.constants[value]; ok {
		return node
	}
	node := g.NewVariableWithName(mat.NewScalar(value), false, fmt.Sprint(value))
	g.constants[value] = node
	return node
}

// NewWrap creates a new wrapper Node for the given value, attaching it to
// the graph.
func (g *Graph[T]) NewWrap(value GradValue[T]) Node[T] {
	n := &Wrapper[T]{
		GradValue: value,
		timeStep:  g.curTimeStep,
		graph:     g,
		wrapGrad:  true,
	}
	return g.insert(n)
}

// NewWrapNoGrad is similar to NewWrap, but it disables automatic
// differentiation on the new node.
func (g *Graph[T]) NewWrapNoGrad(value GradValue[T]) Node[T] {
	n := &Wrapper[T]{
		GradValue: value,
		graph:     g,
		timeStep:  g.curTimeStep,
		wrapGrad:  false,
	}
	return g.insert(n)
}

// nodeInternal extends the public Node with private methods.
type nodeInternal[T mat.DType] interface {
	Node[T]
	setID(int)
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

// releaseMemory clears the values and the gradients of operator nodes.
// Since the values and the gradients within the nodes are handled through a pool of dense matrices,
// releasing them allows the memory to be reused without being reallocated, improving performance.
// By setting retain graph to false, the operators are freed and thus the graph is disintegrated.
func (g *Graph[T]) releaseMemory(retainGraph bool) {
	g.fWG.Wait()
	g.bWG.Wait()
	for _, node := range g.nodes {
		op, ok := node.(*Operator[T])
		if !ok {
			continue
		}
		op.releaseValue()
		op.releaseGrad()
		if retainGraph {
			op.valueMx.TryLock()
			if op.gradsMx == nil {
				op.gradsMx = new(sync.RWMutex)
			}
			op.gradsMx.TryLock()
		} else {
			// free operator
			*op = Operator[T]{}
			getOperatorPool[T]().Put(op)
			// TODO: release constants?
		}
	}
}

func (g *Graph[T]) nodeBoundaries(fromTimeStep, toTimeStep int) (startNodeIndex, endNodeIndex int) {
	startNodeIndex = g.timeStepBoundaries[fromTimeStep] // inclusive
	endNodeIndex = len(g.nodes)                         // exclusive

	if toTimeStep != -1 && toTimeStep != g.TimeStep() {
		endNodeIndex = g.timeStepBoundaries[toTimeStep+1]
	}
	return
}

func anyNodeRequiresGrad[T mat.DType](nodes []Node[T]) bool {
	for _, node := range nodes {
		if node.RequiresGrad() {
			return true
		}
	}
	return false
}

// MarshalBinary satisfies encoding.BinaryMarshaler interface and prevents
// a Graph to be encoded to binary representation.
// This is relevant in the context of a Graph being part of a nn.Model: when
// serializing a model to binary, we want to skip the Graph, since it is part
// of the runtime context only.
func (g *Graph[_]) MarshalBinary() ([]byte, error) {
	return []byte{}, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
func (g *Graph[_]) UnmarshalBinary(_ []byte) error {
	return nil
}
