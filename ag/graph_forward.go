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

	// Free the values that are about to be recalculated so that memory is not wasted
	for _, node := range g.nodes {
		if op, ok := node.(*Operator[T]); ok {
			if op.timeStep >= handler.fromTimeStep && (handler.toTimeStep == -1 || op.timeStep <= handler.toTimeStep) {
				g.releaseValue(op)
			}
		}
	}

	if g.maxProc > 1 {
		handler.runConcurrent()
	} else {
		handler.runSerial()
	}
}

type forwardHandler[T mat.DType] struct {
	g            *Graph[T]
	fromTimeStep int // default 0
	toTimeStep   int // default -1 (no limit)
}

func (h *forwardHandler[T]) runSerial() {
	for _, node := range h.g.nodes {
		if op, ok := node.(*Operator[T]); ok {
			if op.timeStep < h.fromTimeStep {
				continue
			}
			if h.toTimeStep != -1 && op.timeStep > h.toTimeStep {
				continue
			}
			op.forward()
		}
	}
}

func (h *forwardHandler[T]) runConcurrent() {
	fromTS, toTS := h.fromTimeStep, h.toTimeStep
	groups := h.g.groupNodesByHeight()

	pqSize := h.g.maxProc
	workCh := make(chan *Operator[T], pqSize)

	allWorkDone := false

	var wg sync.WaitGroup

	for i := 0; i < pqSize; i++ {
		go func() {
			for !allWorkDone {
				select {
				case op := <-workCh:
					if op == nil {
						continue
					}
					op.forward()
					wg.Done()
				}
			}
		}()
	}

	for _, group := range groups {
		for _, node := range group {
			op, isOperator := node.(*Operator[T])
			if !isOperator || (op.timeStep < fromTS || (toTS != -1 && op.timeStep > toTS)) {
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
