// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"sync"
)

type forwardHandler struct {
	g            *Graph
	fromTimeStep int64 // default 0
	toTimeStep   int64 // default -1 (no limit)
}

// TODO: check time-step range
func (f *forwardHandler) runSerial() {
	for _, node := range f.g.nodes {
		if node, ok := node.(*operator); ok {
			node.value = node.function.Forward()
		}
	}
}

// TODO: check time-step range
func (f *forwardHandler) runConcurrent() {
	groups := f.g.groupNodesByHeight()
	var wg sync.WaitGroup
	for _, group := range groups {
		for _, node := range group {
			if op, ok := node.(*operator); ok {
				wg.Add(1)
				go func(op *operator) {
					defer wg.Done()
					op.value = op.function.Forward()
				}(op)
			}
		}
		wg.Wait()
	}
}

type backwardHandler struct {
	g              *Graph
	node           Node
	outputGrad     mat.Matrix
	stopAtTimeStep int64 // default -1 (full backward)
}

func (f *backwardHandler) propagateOutputGrad() {
	gx := f.outputGrad
	if gx == nil {
		gx = f.node.Value().OnesLike()
		defer mat.ReleaseDense(gx.(*mat.Dense))
	}
	f.node.PropagateGrad(gx)
}

func (f *backwardHandler) runSerial() {
	nodes := f.g.nodes
	lastIndex := f.node.Id()
	stopAtTimeStep := f.stopAtTimeStep
	truncated := stopAtTimeStep > -1
	_ = nodes[lastIndex] // avoid bounds check
	for i := lastIndex; i >= 0; i-- {
		if truncated && nodes[i].getTimeStep() <= stopAtTimeStep {
			break
		}
		if node, ok := nodes[i].(*operator); ok {
			node.backward()
		}
	}
}

func (f *backwardHandler) runConcurrent() {
	stopAtTimeStep := f.stopAtTimeStep
	truncated := stopAtTimeStep > -1
	groups := f.g.groupNodesByHeight()
	lastGroupIndex := f.g.cache.height[f.node.Id()]
	lastNodeIndex := f.node.Id()
	var wg sync.WaitGroup
	for i := lastGroupIndex; i >= 0; i-- {
		for _, node := range groups[i] {
			if truncated && node.getTimeStep() <= stopAtTimeStep {
				break
			}
			if op, ok := node.(*operator); ok {
				if op.id > lastNodeIndex {
					continue
				}
				wg.Add(1)
				go func(op *operator) {
					defer wg.Done()
					op.backward()
				}(op)
			}
		}
		wg.Wait()
	}
}
