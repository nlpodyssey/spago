// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "log"

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
	for _, node := range g.nodes[start:end] {
		if op, ok := node.(*Operator[T]); ok {
			if op.value != nil {
				continue
			}
			op.valueAtomicFlag = 0
			op.valueMx.TryLock()
			g.fWG.Add(1)
			go op.forward()
		}
	}
	g.fWG.Wait()
}
