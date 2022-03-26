// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"log"
	"sync"
)

// Forward computes the results of the entire Graph (not thread-safe).
// Usually you don't need to execute Forward() manually in define-by-run configuration (default).
// The method is equivalent to calling ForwardT with a range [0, -1].
func (g *Graph[T]) Forward() {
	g.ForwardT(0, -1)
}

// ForwardT is a truncated version of Forward that computes only a portion of nodes within a time-step range.
// [0, -1] equals computing the entire graph.
// Not thread-safe.
func (g *Graph[T]) ForwardT(fromTimeStep, toTimeStep int) {
	if fromTimeStep < 0 {
		log.Fatalf("ag: expected fromTimeStep equal to or greater than zero. Found %d.", fromTimeStep)
	}
	if toTimeStep > -1 && toTimeStep < fromTimeStep {
		log.Fatalf("ag: expected toTimeStep equal to or greater than `%d` (fromTimeStep). Found `%d`.",
			fromTimeStep, toTimeStep)
	}
	start, end := g.nodeBoundaries(fromTimeStep, toTimeStep)
	g.forward(start, end)
}

func (g *Graph[T]) forward(start, end int) {
	if g.maxProc > 1 {
		g.forwardConcurrent(start, end)
		return
	}
	g.forwardSerial(start, end)
}

func (g *Graph[T]) forwardSerial(start, end int) {
	for _, node := range g.nodes[start:end] {
		if op, ok := node.(*Operator[T]); ok {
			g.releaseValue(op)
			op.forward()
		}
	}
	g.cache.lastComputedID = end
}

func (g *Graph[T]) forwardConcurrent(start, end int) {
	pending := make([]chan struct{}, len(g.nodes[start:end]))
	for i := range pending {
		if _, ok := g.nodes[i+start].(*Operator[T]); !ok {
			continue
		}
		pending[i] = make(chan struct{})
	}

	var wg sync.WaitGroup

	workCh := make(chan struct{}, g.maxProc)

	forward := func(node *Operator[T]) {
		defer wg.Done()

		// All operands must be resolved before proceeding
		for _, operand := range node.Operands() {
			idx := operand.ID() - start
			if operand.ID() < start || pending[idx] == nil {
				continue
			}
			<-pending[idx] // wait until the node value is available
		}

		workCh <- struct{}{}
		node.forward()
		close(pending[node.id-start]) // broadcast node resolution
		<-workCh
	}

	for _, node := range g.nodes[start:end] {
		if operator, ok := node.(*Operator[T]); ok {
			wg.Add(1)
			go forward(operator)
		}
	}
	wg.Wait()

	for i := 0; i < g.maxProc; i++ {
		workCh <- struct{}{}
	}
	close(workCh)

	g.cache.lastComputedID = end
}
