// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/mat"

// Backward starts the back-propagation from the node.
//
// It follows these mutually exclusive rules:
//   a) the node has gradients already (probably assigned externally via node.PropagateGrads()), use those;
//   b) the output gradients are passed, use those;
//   c) the output gradients are automatically assigned by finding the derivative of the node with respect
//      to the node itself (dy/dy = 1).
//
// The gradients, except those of the given node, are summed to the existing ones during the process.
// Unless that's what you want, make sure all nodes have zero gradients.
//
// It panics if gradients are passed but the node already has them assigned.
func Backward[T mat.DType](node Node[T], grad ...mat.Matrix[T]) {
	if grad != nil && node.HasGrad() {
		panic("ag: attempt to start a backward with output gradients on a node that already has gradients.")
	}
	if !node.HasGrad() {
		var gx mat.Matrix[T]
		if len(grad) == 0 || grad[0] == nil {
			gx = node.Value().OnesLike()
			defer mat.ReleaseMatrix(gx)
		} else {
			gx = grad[0]
		}
		node.PropagateGrad(gx)
	}

	node.Graph().backward(node.ID(), 0)
}

// BackwardT is the same as Backward but ends back-propagation on the first node with a time-step
// that is less or equal to the number of back steps.
func BackwardT[T mat.DType](node Node[T], backSteps int, grad ...mat.Matrix[T]) {
	if grad != nil && node.HasGrad() {
		panic("ag: attempt to start a backward with output gradients on a node that already has gradients.")
	}
	if !node.HasGrad() {
		var gx mat.Matrix[T]
		if len(grad) == 0 || grad[0] == nil {
			gx = node.Value().OnesLike()
			defer mat.ReleaseMatrix(gx)
		} else {
			gx = grad[0]
		}
		node.PropagateGrad(gx)
	}

	node.Graph().backward(node.ID(), node.Graph().timeStepBoundaries[node.TimeStep()-backSteps])
}

// Backward performs full back-propagation from the last node of the graph.
// It requires the root nodes to have assigned gradients already.
//
// It visits each node in reverse topological order, to propagate the gradients
// from the given node all the way back to the leaf.
//
// Note that the gradients are summed to the existing ones.
// Unless that's what you want, make sure  all nodes have zero gradients.
func (g *Graph[T]) Backward() {
	g.backward(g.maxID, 0)
}

// BackwardT performs Truncated Back-Propagation Through Time.
// It is the same as Backward but ends back-propagation on the first node with a time-step
// that is less or equal to the number of back steps.
// The TBTT can perform without the need to recalculate the values of previous nodes (Williams and Peng, 1990).
func (g *Graph[T]) BackwardT(backSteps int) {
	g.backward(g.maxID, g.timeStepBoundaries[g.nodes[g.maxID].TimeStep()-backSteps])
}

func (g *Graph[T]) backward(start, end int) {
	g.fWG.Wait()
	g.backwardInProgress = true
	defer func() {
		g.backwardInProgress = false
	}()
	nodes := g.nodes
	_, _ = nodes[start], nodes[end] // avoid bounds check
	for i := start; i >= end; i-- {
		if op, ok := nodes[i].(*Operator[T]); ok {
			g.bWG.Add(1)
			go op.backward()
		}
	}
	g.bWG.Wait()
}
