// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "sync"

// Backward initiates back-propagation from the input nodes.
//
// The function operates according to the following mutually exclusive rules:
//   - If the node already has gradients (likely assigned externally via node.AccGrads()), those gradients are used.
//   - If the node does not have gradients assigned, the output gradients are automatically assigned by finding the derivative of the node with respect to itself (dy/dy = 1).
//
// During the back-propagation process, the gradients of all nodes, except for the given node, are summed to the existing gradients.
// Unless you intend to do so, ensure that all nodes have zero gradients.
//
// The function returns ReleaseGraph as ReleaseGraphFunc.
func Backward(xs ...Node) ReleaseGraphFunc {
	ops := filterOperators(xs)
	if len(ops) == 0 {
		// There are no operators to process, do nothing.
		return nil
	}

	prepareBackwardPass(ops)
	setOutputGrads(ops)
	processBackwardPass(ops)

	return func() {
		ReleaseGraph(xs...)
	}
}

// filterOperators returns a list of operators from a list of nodes.
func filterOperators(nodes []Node) []*Operator {
	operators := make([]*Operator, 0, len(nodes))
	for _, node := range nodes {
		if op, ok := node.(*Operator); ok {
			operators = append(operators, op)
		}
	}
	return operators
}

// prepareBackwardPass prepares the operators for the backward pass.
func prepareBackwardPass(operators []*Operator) {
	for _, op := range operators {
		op.prepareBackwardPass()
	}
}

// setOutputGrads initializes the output gradients for the operators.
func setOutputGrads(operators []*Operator) {
	for _, op := range operators {
		op.setOutputGrad() // Implement this method in Operator
	}
}

// processBackwardPass executes the backward pass for the operators.
func processBackwardPass(operators []*Operator) {
	wg := &sync.WaitGroup{}
	for _, op := range operators {
		op.processBackwardPass(wg)
	}
	wg.Wait()
}
