// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"log"
	"sync"
	"sync/atomic"
)

type Graph struct {
	// to avoid data race during concurrent computations
	mu sync.Mutex
	// maxId is the id of the last inserted node (corresponds of len(nodes)-1)
	maxId int64
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	nodes []Node
	// randGen is the generator of random numbers
	randGen *rand.LockedRand
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.Rand.
func NewGraph(opt ...interface{}) *Graph {
	g := &Graph{maxId: 0, nodes: nil}

	for _, t := range opt {
		switch t := t.(type) {
		case *rand.LockedRand:
			g.randGen = t
		default:
			log.Fatal("graph: invalid init options")
		}
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
	g.maxId = 0
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

// NewVariable creates e returns a new node.
func (g *Graph) NewVariable(value mat.Matrix, requiresGrad bool) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &variable{
		graph:        g,
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
		id:        g.newId(),
		wrapGrad:  false,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, newNode)
	return newNode
}

func (g *Graph) ForwardAll() {
	for _, node := range g.nodes {
		if node, ok := node.(*operator); ok {
			node.value = node.function.Forward()
		}
	}
}

// Backward visit each node in reverse topological order, to propagate the gradients from the given node all the way
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
	nodes := g.nodes
	lastIndex := node.Id()
	_ = nodes[lastIndex] // avoid bounds check
	for i := lastIndex; i >= 0; i-- {
		if node, ok := nodes[i].(*operator); ok {
			node.backward()
		}
	}
}

func (g *Graph) BackwardAll() {
	nodes := g.nodes
	lastIndex := len(nodes) - 1
	_ = nodes[lastIndex]
	for i := lastIndex; i >= 0; i-- {
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

// newId generates and returns a new incremental sequential ID.
func (g *Graph) newId() int64 {
	return atomic.AddInt64(&g.maxId, 1) - 1
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
