// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/saientist/spago/pkg/mat"
	"github.com/saientist/spago/pkg/ml/ag/fn"
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
}

type nodeInfo struct {
	node Node
	// depth is the maximum depth reached by the node/.
	depth int
	// descendants contains the ids of all descendants including the node itself.
	descendants []int64
}

// NewGraph returns a new initialized graph.
func NewGraph() *Graph {
	return &Graph{
		maxId:    0,
		maxDepth: 0,
		nodes:    make([]*nodeInfo, 0),
	}
}

func (g *Graph) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.maxId = 0
	g.maxDepth = 0
	g.nodes = make([]*nodeInfo, 0)
}

// NewVariable creates e returns a new node.
func (g *Graph) NewVariable(value mat.Matrix, requiresGrad bool) *Variable {
	g.mu.Lock()
	defer g.mu.Unlock()
	newId := g.newId()
	newNode := &Variable{
		graph:        g,
		id:           newId,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:        newNode,
		depth:       0,
		descendants: []int64{newId},
	})
	return newNode
}

// NewScalar creates a variable node that doesn't require gradients
func (g *Graph) NewScalar(value float64) *Variable {
	return g.NewVariable(mat.NewScalar(value), false)
}

// NewOperator creates a new operator along with its forward pass.
func (g *Graph) NewOperator(f fn.Function, operands ...Node) *Operator {
	value := f.Forward() // the calculation can be concurrent
	g.mu.Lock()
	defer g.mu.Unlock()
	newId := g.newId()
	newNode := &Operator{
		graph:        g,
		id:           newId,
		function:     f,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requireGrad(operands),
	}

	descendants := make([]int64, 0, g.sumDescendants(operands)+1) // + itself
	mark := make([]bool, len(g.nodes), len(g.nodes))
	for _, o := range operands {
		for _, descendantId := range g.nodes[o.Id()].descendants {
			if !mark[descendantId] {
				mark[descendantId] = true
				g.nodes[descendantId].depth++
				g.maxDepth = maxDepth(g.nodes[descendantId].depth, g.maxDepth)
				descendants = append(descendants, descendantId)
			}
		}
	}
	descendants = append(descendants, newId)

	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:        newNode,
		depth:       0,
		descendants: descendants,
	})
	return newNode
}

func (g *Graph) NewWrap(value GradValue) *Wrapper {
	g.mu.Lock()
	defer g.mu.Unlock()
	newId := g.newId()
	newNode := &Wrapper{
		GradValue: value,
		graph:     g,
		id:        newId,
		wrapGrad:  true,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:        newNode,
		depth:       0,
		descendants: []int64{newId},
	})
	return newNode
}

func (g *Graph) NewWrapNoGrad(value GradValue) *Wrapper {
	g.mu.Lock()
	defer g.mu.Unlock()
	newId := g.newId()
	newNode := &Wrapper{
		GradValue: value,
		graph:     g,
		id:        newId,
		wrapGrad:  false,
	}
	// the new id is sequential so this the append is fine
	g.nodes = append(g.nodes, &nodeInfo{
		node:        newNode,
		depth:       0,
		descendants: []int64{newId},
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
				if x, ok := x.(*Operator); ok {
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
	} else {
		gx = grad[0]
	}
	node.PropagateGrad(gx)
	minDepth := g.nodes[node.Id()].depth
	for depth, ns := range g.groupNodesByDepth() {
		if depth < minDepth {
			break
		}
		var wg sync.WaitGroup
		wg.Add(len(ns))
		for _, n := range ns {
			go func(x Node) {
				defer wg.Done()
				if x, ok := x.(*Operator); ok {
					x.backward()
				}
			}(n)
		}
		wg.Wait()
	}
}

func (g *Graph) BackwardAll() {
	for _, ns := range g.groupNodesByDepth() {
		for _, n := range ns {
			if n, ok := n.(*Operator); ok {
				n.backward()
			}
		}
	}
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

func maxDepth(a, b int) int {
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
