// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"sync"
)

type forwardHandler struct {
	g            *Graph
	fromTimeStep int // default 0
	toTimeStep   int // default -1 (no limit)
}

func (h *forwardHandler) runSerial() {
	for _, node := range h.g.nodes {
		if op, ok := node.(*Operator); ok {
			if op.timeStep < h.fromTimeStep {
				continue
			}
			if h.toTimeStep != -1 && op.timeStep > h.toTimeStep {
				continue
			}
			op.value = op.function.Forward()
		}
	}
}

func (h *forwardHandler) runConcurrent() {
	fromTS, toTS := h.fromTimeStep, h.toTimeStep
	groups := h.g.groupNodesByHeight()

	var wg sync.WaitGroup
	for _, group := range groups {
		for _, node := range group {
			op, isOperator := node.(*Operator)
			if !isOperator || (op.timeStep < fromTS || (toTS != -1 && op.timeStep > toTS)) {
				continue
			}
			wg.Add(1)
			h.g.processingQueue.Go(func() {
				defer wg.Done()
				op.value = op.function.Forward()
			})
		}
		wg.Wait()
	}
}

type backwardHandler struct {
	g              *Graph
	node           Node
	outputGrad     mat.Matrix
	stopAtTimeStep int // default -1 (full backward)
}

func (h *backwardHandler) propagateOutputGrad() {
	gx := h.outputGrad
	if gx == nil {
		gx = h.node.Value().OnesLike()
		defer mat.ReleaseMatrix(gx)
	}
	h.node.PropagateGrad(gx)
}

func (h *backwardHandler) runSerial() {
	nodes := h.g.nodes
	lastIndex := h.node.ID()
	stopAtTimeStep := h.stopAtTimeStep
	truncated := stopAtTimeStep > -1
	_ = nodes[lastIndex] // avoid bounds check
	for i := lastIndex; i >= 0; i-- {
		if truncated && nodes[i].TimeStep() <= stopAtTimeStep {
			break
		}
		if node, ok := nodes[i].(*Operator); ok {
			node.backward()
		}
	}
}

func (h *backwardHandler) runConcurrent() {
	stopAtTimeStep := h.stopAtTimeStep
	truncated := stopAtTimeStep > -1
	groups := h.g.groupNodesByHeight()
	lastGroupIndex := h.g.cache.height[h.node.ID()]
	lastNodeIndex := h.node.ID()
	var wg sync.WaitGroup
	for i := lastGroupIndex; i >= 0; i-- {
		for _, node := range groups[i] {
			if truncated && node.TimeStep() <= stopAtTimeStep {
				break
			}
			op, isOperator := node.(*Operator)
			if !isOperator {
				continue
			}
			if op.id > lastNodeIndex {
				continue
			}
			wg.Add(1)
			h.g.processingQueue.Go(func() {
				defer wg.Done()
				op.backward()
			})
		}
		wg.Wait()
	}
}
