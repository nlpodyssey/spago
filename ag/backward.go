// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "sync"

// Backward initiates back-propagation from the input nodes.
//
// The function operates according to the following mutually exclusive rules:
//   - If the node already has gradients (likely assigned externally via node.AccGrads()), those gradients are used.
//   - If the node does not have gradients assigned and is a scalar, the output gradients are automatically assigned
//     by finding the derivative of the node with respect to itself (dy/dy = 1).
//   - If the node does not have gradients assigned and is not a scalar, it returns an error.
//
// During the back-propagation process, the gradients of all nodes, except for the given node, are summed to the existing gradients.
// Unless you intend to do so, ensure that all nodes have zero gradients.
func Backward(xs ...DualValue) error {
	ops := filterOperators(xs)
	if len(ops) == 0 {
		return nil
	}

	// The three for loops below are intentionally executed in sequence.

	// 1. Prepare the backward pass for each operator.
	for _, op := range ops {
		op.prepareBackwardPass()
	}

	// 2. Assign the output gradients for each operator.
	for _, op := range ops {
		if err := op.assignOutputGradient(); err != nil {
			return err
		}
	}

	// 3. Process the backward pass for each operator in parallel using wait groups.
	// These steps must occur in this order, so the loops cannot be combined due to their sequential dependencies.
	wg := &sync.WaitGroup{}
	for _, op := range ops {
		op.processBackwardPass(wg)
	}
	wg.Wait()

	return nil
}

// filterOperators returns a list of operators from a list of nodes.
func filterOperators(nodes []DualValue) []*Operator {
	ops := make([]*Operator, 0, len(nodes))
	for _, node := range nodes {
		switch op := node.(type) {
		case *Operator:
			ops = append(ops, op)
		}
	}
	return ops
}
