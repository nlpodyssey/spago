// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"log"
	"sync"
)

// ForwardOption allows to adapt the Forward() to your specific needs.
type ForwardOption[T mat.DType] func(*forwardHandler[T])

// Range allows you to limit the forward computation within a time-step range.
// By default, the forward computes from the first node at time-step 0 to the last node at the current time-step.
func Range[T mat.DType](fromTimeStep, toTimeStep int) ForwardOption[T] {
	if fromTimeStep < 0 {
		log.Fatalf("ag: expected fromTimeStep equal to or greater than zero. Found %d.", fromTimeStep)
	}
	if toTimeStep > -1 && toTimeStep < fromTimeStep {
		log.Fatalf("ag: expected toTimeStep equal to or greater than `%d` (fromTimeStep). Found `%d`.",
			fromTimeStep, toTimeStep)
	}
	return func(f *forwardHandler[T]) {
		f.fromTimeStep = fromTimeStep
		f.toTimeStep = toTimeStep
	}
}

// Forward computes the results of the entire Graph.
// Usually you don't need to execute Forward() manually in the define-by-run configuration (default).
// If you do, all values will be recalculated. You can also choose through the Range option to recalculate only a portion of nodes.
// Instead, it is required to obtain the value of the nodes in case the Graph has been created with WithEagerExecution(false).
func (g *Graph[T]) Forward(opts ...ForwardOption[T]) {
	handler := &forwardHandler[T]{
		g:            g,
		fromTimeStep: 0,
		toTimeStep:   -1, // unlimited
	}
	for _, opt := range opts {
		opt(handler)
	}
	if g.maxProc > 1 {
		handler.runConcurrent()
		return
	}
	handler.runSerial()
}

type forwardHandler[T mat.DType] struct {
	g            *Graph[T]
	fromTimeStep int // default 0
	toTimeStep   int // default -1 (no limit)
}

func (h *forwardHandler[T]) runSerial() {
	offset, end := h.nodeBoundaries()
	for _, node := range h.g.nodes[offset:end] {
		if op, ok := node.(*Operator[T]); ok {
			h.g.releaseValue(op)
			op.forward()
		}
	}
}

func (h *forwardHandler[T]) runConcurrent() {
	offset, end := h.nodeBoundaries()

	pending := make([]chan struct{}, len(h.g.nodes[offset:end]))
	for i := range pending {
		if _, ok := h.g.nodes[i+offset].(*Operator[T]); !ok {
			continue // skip elements have nil value
		}
		pending[i] = make(chan struct{})
	}

	var wg sync.WaitGroup

	workCh := make(chan struct{}, h.g.maxProc)

	forward := func(node *Operator[T]) {
		defer wg.Done()

		// All operands must be resolved before proceeding
		for _, operand := range node.Operands() {
			if operand.ID() < offset {
				continue
			}
			idx := operand.ID() - offset
			if pending[idx] == nil {
				continue
			}
			<-pending[idx] // wait until the node value is available
		}

		workCh <- struct{}{}
		h.g.releaseValue(node)
		node.forward()
		close(pending[node.id-offset]) // broadcast node resolution
		<-workCh
	}

	for _, node := range h.g.nodes[offset:end] {
		if operator, ok := node.(*Operator[T]); ok {
			wg.Add(1)
			go forward(operator)
		}
	}
	wg.Wait()

	for i := 0; i < h.g.maxProc; i++ {
		workCh <- struct{}{}
	}
	close(workCh)
}

func (h *forwardHandler[T]) nodeBoundaries() (start, end int) {
	start = h.g.timeStepBoundaries[h.fromTimeStep] // inclusive
	end = len(h.g.nodes)                           // exclusive

	if h.toTimeStep != -1 && h.toTimeStep != h.g.TimeStep() {
		end = h.g.timeStepBoundaries[h.toTimeStep+1]
	}
	return
}
