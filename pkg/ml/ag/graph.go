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
	// the maximum depth reached by a node of the graph
	maxDepth int
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	nodes []*nodeInfo
	// randGen is the generator of random numbers
	randGen *rand.LockedRand
}

type nodeInfo struct {
	node Node
	// depth is the maximum depth reached by the node.
	depth int
	// the id of the last operator visiting this node for depth calculation.
	lastVisitorId int64
	// descendants contains the ids of all descendants including the node itself.
	descendants []int64
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.Rand.
func NewGraph(opt ...interface{}) *Graph {
	g := &Graph{
		maxId:    0,
		maxDepth: 0,
		nodes:    nil,
	}

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
// It releases the matrices underlying the nodes so to reduce the need of future new time-consuming allocations.
// It is important to stress that calling g.Clean(), the "value" and "grad" matrices of the operators nodes are freed (set to nil).
// Whoever is using the Value() or Grad() properties of a node, does so at his own risk. It is therefore recommended
// to make always a copy of the return value of Value() or Grad().
// Alternatively, you can use the convenient graph's methods g.GetCopiedValue(node) and g.GetCopiedGrad(node).
func (g *Graph) Clear() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.maxId = 0
	g.maxDepth = 0
	g.releaseMemory()
	g.nodes = nil
}

// releaseMemory clears the values and the gradients of operator nodes.
// Since the values and the gradients within the nodes are handled through a pool of dense matrices,
// releasing them allows the memory to be reused without being reallocated, improving performance.
func (g *Graph) releaseMemory() {
	for _, item := range g.nodes {
		if node, ok := item.node.(*operator); ok {
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
	if node.grad == nil {
		return
	}
	node.ZeroGrad()
}

// NewVariable creates e returns a new node.
func (g *Graph) NewVariable(value mat.Matrix, requiresGrad bool) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newId := g.newId()
	newNode := &variable{
		graph:        g,
		id:           newId,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:          newNode,
		depth:         0,
		descendants:   []int64{newId},
		lastVisitorId: -1,
	})
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
	newId := g.newId()
	newNode := &operator{
		graph:        g,
		id:           newId,
		function:     f,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requireGrad(operands),
	}

	descendants := make([]int64, 0, g.sumDescendants(operands)+1) // + itself
	for _, o := range operands {
		for _, descendantId := range g.nodes[o.Id()].descendants {
			descendantNode := g.nodes[descendantId]
			if descendantNode.lastVisitorId == newId {
				continue
			}
			descendantNode.lastVisitorId = newId
			descendantNode.depth++
			g.maxDepth = max(descendantNode.depth, g.maxDepth)
			descendants = append(descendants, descendantId)
		}
	}
	descendants = append(descendants, newId)

	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:          newNode,
		depth:         0,
		descendants:   descendants,
		lastVisitorId: -1,
	})
	return newNode
}

func (g *Graph) NewWrap(value GradValue) *wrapper {
	g.mu.Lock()
	defer g.mu.Unlock()
	newId := g.newId()
	newNode := &wrapper{
		GradValue: value,
		graph:     g,
		id:        newId,
		wrapGrad:  true,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:          newNode,
		depth:         0,
		descendants:   []int64{newId},
		lastVisitorId: -1,
	})
	return newNode
}

func (g *Graph) NewWrapNoGrad(value GradValue) *wrapper {
	g.mu.Lock()
	defer g.mu.Unlock()
	newId := g.newId()
	newNode := &wrapper{
		GradValue: value,
		graph:     g,
		id:        newId,
		wrapGrad:  false,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:          newNode,
		depth:         0,
		descendants:   []int64{newId},
		lastVisitorId: -1,
	})
	return newNode
}

func (g *Graph) ForwardAll() {
	groups := g.groupNodesByDepth()
	for depth := len(groups) - 1; depth >= 0; depth-- {
		var wg sync.WaitGroup
		wg.Add(len(groups[depth]))
		for _, n := range groups[depth] {
			go func(x Node) {
				defer wg.Done()
				x.ZeroGrad()
				if x, ok := x.(*operator); ok {
					x.value = x.function.Forward()
				}
			}(n)
		}
		wg.Wait()
	}
}

// Backward propagates the gradients from the node all the way back to the leaf descendants i.e. variables.
// If there are no input gradients (i.e. grad is nil), it starts by finding the derivative of the final output
// with respect to the final output itself.
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
	minDepth := g.nodes[node.Id()].depth
	nodesPerDepth := g.groupNodesByDepth()
	for depth, ns := range nodesPerDepth {
		if depth < minDepth {
			break
		}
		for _, node := range ns {
			if node, ok := node.(*operator); ok {
				node.backward()
			}
		}
	}
}

// Backward propagates the gradients from the node all the way back to the leaf descendants i.e. variables.
// If there are no input gradients (i.e. grad is nil), it starts by finding the derivative of the final output
// with respect to the final output itself.
func (g *Graph) BackwardConcurrent(node Node, grad ...mat.Matrix) {
	var gx mat.Matrix
	if len(grad) > 1 {
		panic("ag: invalid number of arguments. Required zero or one argument.")
	} else if len(grad) == 0 || grad[0] == nil {
		gx = node.Value().OnesLike()
	} else {
		gx = grad[0]
	}
	node.PropagateGrad(gx)
	minDepth := g.nodes[node.Id()].depth
	nodesPerDepth := g.groupNodesByDepth()
	for depth, ns := range nodesPerDepth {
		if depth < minDepth {
			break
		}
		var wg sync.WaitGroup
		wg.Add(len(ns))
		for _, node := range ns {
			go func(x Node) {
				defer wg.Done()
				if x, ok := x.(*operator); ok {
					x.backward()
				}
			}(node)
		}
		wg.Wait()
	}
}

func (g *Graph) BackwardAll() {
	for _, ns := range g.groupNodesByDepth() {
		for _, n := range ns {
			if n, ok := n.(*operator); ok {
				n.backward()
			}
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

func (g *Graph) sumDescendants(ns []Node) int {
	sum := 0
	for _, n := range ns {
		sum += len(g.nodes[n.Id()].descendants)
	}
	return sum
}

// groupNodesByDepth returns the nodes of the graph grouped by depth.
func (g *Graph) groupNodesByDepth() [][]Node {
	out := make([][]Node, g.maxDepth+1)
	for _, n := range g.nodes {
		out[n.depth] = append(out[n.depth], n.node)
	}
	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
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
