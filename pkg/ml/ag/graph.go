// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"sync"
	"sync/atomic"
)

type Graph struct {
	// to avoid data race during concurrent computations
	mu sync.Mutex
	// maxId is the id of the last inserted node (corresponds of len(nodes)-1)
	maxId int64
	// the time-step is useful to perform truncated back propagation (default 0)
	curTimeStep int64
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	nodes []Node
	// randGen is the generator of random numbers
	randGen *rand.LockedRand
}

type GraphOption func(*Graph)

func Rand(rand *rand.LockedRand) GraphOption {
	return func(g *Graph) {
		g.randGen = rand
	}
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.Rand.
func NewGraph(opts ...GraphOption) *Graph {
	g := &Graph{
		maxId:       -1,
		curTimeStep: 0,
		nodes:       nil,
	}
	for _, opt := range opts {
		opt(g)
	}
	if g.randGen == nil {
		g.randGen = rand.NewLockedRand(1) // set default random generator
	}
	return g
}

// Clear cleans the graph. This is a destructive operation.
// It is not mandatory to call this method, but it is strongly recommended to do so when you finish using the graph.
// The cleaning of the graph improves the memory management and therefore the efficiency of execution.
// Clear releases the matrices underlying the nodes so to reduce the need of future new time-consuming allocations.
// It is important to stress that calling g.Clean(), the "value" and "grad" of the operators nodes are freed (set to nil).
// Whoever is using the Value() or Grad() properties of a node, does so at his own risk. It is therefore recommended to
// make always a copy of the return value of Value() or Grad().
// Alternatively, you can use the convenient graph's methods g.GetCopiedValue(node) and g.GetCopiedGrad(node).
func (g *Graph) Clear() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.maxId = -1
	g.curTimeStep = 0
	g.releaseMemory()
	g.nodes = nil
}

// ClearForReuse() does the same thing as Clear(), with the difference that the graph structure i.e. how nodes are
// connected to each other, is maintained.
// This allows you to efficiently use the graph as if it were "pre-computed" (see the ForwardAll() method for this usage).
func (g *Graph) ClearForReuse() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.releaseMemory()
}

// releaseMemory clears the values and the gradients of operator nodes.
// Since the values and the gradients within the nodes are handled through a pool of dense matrices,
// releasing them allows the memory to be reused without being reallocated, improving performance.
func (g *Graph) releaseMemory() {
	for _, node := range g.nodes {
		if node, ok := node.(*operator); ok {
			g.releaseValue(node)
			g.releaseGrad(node)
		}
	}
}

func (g *Graph) releaseValue(node *operator) {
	if node.value == nil {
		return
	}
	mat.ReleaseDense(node.value.(*mat.Dense))
	node.value = nil
}

func (g *Graph) releaseGrad(node *operator) {
	node.ZeroGrad()
}

func (g *Graph) ZeroGrad() {
	for _, node := range g.nodes {
		node.ZeroGrad()
	}
}

// NewVariable creates e returns a new node.
func (g *Graph) NewVariable(value mat.Matrix, requiresGrad bool) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &variable{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           g.newId(),
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewScalar creates a variable node that doesn't require gradients
func (g *Graph) NewScalar(value float64) Node {
	return g.NewVariable(mat.NewScalar(value), false)
}

// NewOperator creates a new operator along with its forward pass.
// Please note that operations must be performed among nodes belonging to the same graph; it panics otherwise.
func (g *Graph) NewOperator(f fn.Function, operands ...Node) Node {
	for _, o := range operands {
		if o.Graph() != g {
			panic("ag: operations cannot be executed among nodes of different graphs. " +
				"You may consider wrapping the nodes you need with NewWrap().")
		}
	}
	value := f.Forward() // the calculation can be concurrent
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &operator{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           g.newId(),
		function:     f,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requireGrad(operands),
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, newNode)
	return newNode
}

func (g *Graph) NewWrap(value GradValue) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &wrapper{
		GradValue: value,
		timeStep:  g.curTimeStep,
		graph:     g,
		id:        g.newId(),
		wrapGrad:  true,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, newNode)
	return newNode
}

func (g *Graph) NewWrapNoGrad(value GradValue) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &wrapper{
		GradValue: value,
		graph:     g,
		timeStep:  g.curTimeStep,
		id:        g.newId(),
		wrapGrad:  false,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, newNode)
	return newNode
}

func (g *Graph) ForwardAll() {
	g.ClearForReuse() // make sure you don't waste memory
	for _, node := range g.nodes {
		if node, ok := node.(*operator); ok {
			node.value = node.function.Forward()
		}
	}
}

// Backward performs the back-propagation.
// It visits each node in reverse topological order, to propagate the gradients from the given node all the way
// back to the leaf. If there are no input gradients (i.e. grad is nil), it starts by finding the derivative of the
// node with respect to the node itself (dy/dy = 1).
func (g *Graph) Backward(node Node, grad ...mat.Matrix) {
	var gx mat.Matrix
	if len(grad) > 1 {
		panic("ag: invalid number of arguments. Required zero or one argument.")
	} else if len(grad) == 0 || grad[0] == nil {
		gx = node.Value().OnesLike()
		defer mat.ReleaseDense(gx.(*mat.Dense))
	} else {
		gx = grad[0]
	}
	node.PropagateGrad(gx)
	g.fullBackPropagation(node)
}

// TBackward performs the truncated back-propagation.
// It visits each node in reverse topological order, to propagate the gradients from the given node all the way
// back to the leaf. The visit ends as soon as it is encountered a node having the time-step less or equal to the
// number of back steps (aka k2).
// If there are no input gradients (i.e. grad is nil), it starts by finding the derivative of the
// node with respect to the node itself (dy/dy = 1).
func (g *Graph) TBackward(node Node, backSteps int, grad ...mat.Matrix) {
	var gx mat.Matrix
	if len(grad) > 1 {
		panic("ag: invalid number of arguments. Required zero or one argument.")
	} else if len(grad) == 0 || grad[0] == nil {
		gx = node.Value().OnesLike()
		defer mat.ReleaseDense(gx.(*mat.Dense))
	} else {
		gx = grad[0]
	}
	node.PropagateGrad(gx)
	if backSteps == -1 { // no limits
		g.fullBackPropagation(node)
	} else {
		g.truncatedBackPropagation(node, backSteps)
	}
}

// BackwardAll performs full back-propagation from the last node of the graph.
func (g *Graph) BackwardAll() {
	g.fullBackPropagation(g.nodes[g.maxId]) // backward from the last node
}

func (g *Graph) fullBackPropagation(node Node) {
	nodes := g.nodes
	lastIndex := node.Id()
	_ = nodes[lastIndex] // avoid bounds check
	for i := lastIndex; i >= 0; i-- {
		if node, ok := nodes[i].(*operator); ok {
			node.backward()
		}
	}
}

func (g *Graph) truncatedBackPropagation(node Node, backSteps int) {
	if node.getTimeStep() != g.curTimeStep {
		panic("ag: the truncated back-propagation must start from a node whose time-step is equal to the current step")
	}
	nodes := g.nodes
	lastIndex := node.Id()
	stopAtTimeStep := g.curTimeStep - int64(backSteps)
	_ = nodes[lastIndex] // avoid bounds check
	for i := lastIndex; i >= 0; i-- {
		if nodes[i].getTimeStep() <= stopAtTimeStep {
			break
		}
		if node, ok := nodes[i].(*operator); ok {
			node.backward()
		}
	}
}

// GetValues returns a copy of the value of a node. If the value is nil, GetCopiedValue returns nil.
// The returned value is a copy, so it is safe to use even after the graph has been cleared calling g.Clear().
// It is important to remember that the Value() property of a Node is a weak access, as the matrix derived from
// graph's operations can be freed.
func (g *Graph) GetCopiedValue(node Node) mat.Matrix {
	if node.Value() == nil {
		return nil
	}
	return node.Value().Clone()
}

// GetValues returns a copy of the gradients of a node. If the gradients are nil, GetCopiedGrad returns nil.
// The returned value is a copy, so it is safe to use even after the graph has been cleared calling g.Clear().
// It is important to remember that the Grad() property of a Node is a weak access, as the matrix derived from
// graph's operations can be freed.
func (g *Graph) GetCopiedGrad(node Node) mat.Matrix {
	if node.Grad() == nil {
		return nil
	}
	return node.Grad().Clone()
}

// ReplaceValue
func (g *Graph) ReplaceValue(node Node, value mat.Matrix) {
	if node, ok := node.(*variable); !ok {
		panic("ag: invalid node. Only variables are allowed to change their value.")
	} else {
		node.value = value
	}
}

func (g *Graph) IncTimeStep() {
	atomic.AddInt64(&g.curTimeStep, 1)
}

func (g *Graph) TimeStep() int {
	return int(g.curTimeStep)
}

// newId generates and returns a new incremental sequential ID.
func (g *Graph) newId() int64 {
	return atomic.AddInt64(&g.maxId, 1)
}

func requireGrad(ns []Node) bool {
	for _, n := range ns {
		if n.RequiresGrad() {
			return true
		}
	}
	return false
}

// operands converts a slice of node to a slice of operands.
func operands(xs []Node) []fn.Operand {
	var out = make([]fn.Operand, len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}
