// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"sync"
)

// BackwardOption allows to adapt the Backward() to your specific needs.
type BackwardOption[T mat.DType] func(*backwardHandler[T])

// Truncate is an option that sets the number of back steps for the
// Truncated Back-Propagation.
func Truncate[T mat.DType](backSteps int) BackwardOption[T] {
	return func(f *backwardHandler[T]) {
		f.stopAtTimeStep = f.node.TimeStep() - backSteps
	}
}

// OutputGrad is an option that sets the output gradients which are the starting
// point for the back-propagation (Backward).
func OutputGrad[T mat.DType](grad mat.Matrix[T]) BackwardOption[T] {
	return func(f *backwardHandler[T]) {
		f.outputGrad = grad
	}
}

// Backward performs the back-propagation.
// If the node is nil, it performs full back-propagation from the last node
// of the graph, requiring the root nodes to have assigned gradients already.
// Options are not considered in this case.
//
// Otherwise, if the node is not nil, the following logic is applied:
//
// It visits each node in reverse topological order, to propagate the gradients from the given node all the way
// back to the leaf. Note that the gradients are summed to the existing ones. Unless that's what you want, make sure
// all nodes have zero gradients.
//
// The back-propagation starts from the node's output gradients, following these mutually exclusive rules:
//   a) the node has gradients (probably assigned externally via node.PropagateGrads()), use those;
//   b) the output gradients are passed through the backward options, use those;
//   c) the output gradients are automatically assigned by finding the derivative of the node with respect
//      to the node itself (dy/dy = 1).
//
// If the optional back steps are set, a Truncated Back-Propagation Through Time is carried out, that is:
// the visit ends as soon as it is encountered a node with time-step less or equal to the number of back steps.
// The TBTT can perform without the need to recalculate the values of previous nodes (Williams and Peng, 1990).
func (g *Graph[T]) Backward(node Node[T], opts ...BackwardOption[T]) {
	if node == nil {
		g.backwardAll()
		return
	}

	if node.Graph() != g {
		panic("ag: backward cannot be executed among nodes of different graphs")
	}
	handler := &backwardHandler[T]{
		g:              g,
		node:           node,
		outputGrad:     nil,
		stopAtTimeStep: -1, // no stop
	}
	for _, opt := range opts {
		opt(handler)
	}
	if !node.HasGrad() {
		handler.propagateOutputGrad()
	}
	if g.maxProc > 1 {
		handler.runConcurrent()
	} else {
		handler.runSerial()
	}
}

// backwardAll performs full back-propagation from the last node of the graph.
// It requires the root nodes to have assigned gradients already.
func (g *Graph[T]) backwardAll() {
	handler := &backwardHandler[T]{
		g:              g,
		node:           g.nodes[g.maxID],
		outputGrad:     nil,
		stopAtTimeStep: -1, // no stop
	}
	if g.maxProc > 1 {
		handler.runConcurrent()
	} else {
		handler.runSerial()
	}
}

type backwardHandler[T mat.DType] struct {
	g              *Graph[T]
	node           Node[T]
	outputGrad     mat.Matrix[T]
	stopAtTimeStep int // default -1 (full backward)
}

func (h *backwardHandler[_]) propagateOutputGrad() {
	gx := h.outputGrad
	if gx == nil {
		gx = h.node.Value().OnesLike()
		defer mat.ReleaseMatrix(gx)
	}
	h.node.PropagateGrad(gx)
}

func (h *backwardHandler[T]) runSerial() {
	nodes := h.g.nodes
	lastIndex := h.node.ID()
	stopAtTimeStep := h.stopAtTimeStep
	truncated := stopAtTimeStep > -1
	_ = nodes[lastIndex] // avoid bounds check
	for i := lastIndex; i >= 0; i-- {
		if truncated && nodes[i].TimeStep() <= stopAtTimeStep {
			break
		}
		if node, ok := nodes[i].(*Operator[T]); ok {
			node.backward()
		}
	}
}

func (h *backwardHandler[T]) runConcurrent() {
	stopAtTimeStep := h.stopAtTimeStep
	truncated := stopAtTimeStep > -1
	groups := h.g.groupNodesByHeight()
	lastGroupIndex := h.g.cache.height[h.node.ID()]
	lastNodeIndex := h.node.ID()

	workCh := make(chan *Operator[T], h.g.maxProc)
	allWorkDone := false

	var wg sync.WaitGroup

	for i := 0; i < h.g.maxProc; i++ {
		go func() {
			for !allWorkDone {
				select {
				case op := <-workCh:
					if op == nil {
						continue
					}
					op.backward()
					wg.Done()
				}
			}
		}()
	}

	for i := lastGroupIndex; i >= 0; i-- {
		for _, node := range groups[i] {
			if truncated && node.TimeStep() <= stopAtTimeStep {
				break
			}
			op, isOperator := node.(*Operator[T])
			if !isOperator {
				continue
			}
			if op.id > lastNodeIndex {
				continue
			}
			wg.Add(1)
			workCh <- op
		}
		wg.Wait()
	}
	allWorkDone = true
	close(workCh)
}
