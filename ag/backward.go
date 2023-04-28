// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "sync"

// operators is a list of operators.
type operators []*Operator

// Backward initiates back-propagation from the input nodes.
//
// The function operates according to the following mutually exclusive rules:
//   - If the node already has gradients (likely assigned externally via node.AccGrads()), those gradients are used.
//   - If the node does not have gradients assigned, the output gradients are automatically assigned by finding the derivative of the node with respect to itself (dy/dy = 1).
//
// During the back-propagation process, the gradients of all nodes, except for the given node, are summed to the existing gradients.
// Unless you intend to do so, ensure that all nodes have zero gradients.
func Backward(xs ...DualValue) {
	ops := filterOperators(xs)
	if len(ops) == 0 {
		// There are no operators to process, do nothing.
		return
	}
	ops.prepareBackwardPass()
	ops.setOutputGrads()
	ops.processBackwardPass()
}

// filterOperators returns a list of operators from a list of nodes.
func filterOperators(nodes []DualValue) operators {
	ops := make([]*Operator, 0, len(nodes))
	for _, node := range nodes {
		if op, ok := node.(*Operator); ok {
			ops = append(ops, op)
		}
	}
	return ops
}

func (ops operators) prepareBackwardPass() {
	for _, op := range ops {
		op.prepareBackwardPass()
	}
}

func (ops operators) setOutputGrads() {
	for _, op := range ops {
		op.setOutputGrad()
	}
}

func (ops operators) processBackwardPass() {
	wg := &sync.WaitGroup{}
	for _, op := range ops {
		op.processBackwardPass(wg)
	}
	wg.Wait()
}
