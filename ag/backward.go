// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// Backward starts the back-propagation from the node.
//
// It follows these mutually exclusive rules:
//
//	a) the node has gradients already (probably assigned externally via node.AccGrads()), use those;
//	b) the output gradients are passed, use those;
//	c) the output gradients are automatically assigned by finding the derivative of the node with respect
//	   to the node itself (dy/dy = 1).
//
// The gradients, except those of the given node, are summed to the existing ones during the process.
// Unless that's what you want, make sure all nodes have zero gradients.
//
// It panics if gradients are passed but the node already has them assigned.
//
// It returns ReleaseGraph as ReleaseGraphFunc.
func Backward(x Node, grad ...mat.Matrix) ReleaseGraphFunc {
	validateGradInput(grad)

	op, ok := x.(*Operator)
	if !ok {
		return nil
	}

	op.prepareBackwardPass()
	op.initOutputGrad(getOutputGrad(grad))

	wg := &sync.WaitGroup{}
	op.executeBackwardPass(wg)
	wg.Wait()

	return func() {
		ReleaseGraph(x)
	}
}

// BackwardMany performs the backpropagation from a list of nodes.
//
// This is particularly useful when there are previously assigned
// gradients on root nodes or when there are multiple distinct losses.
//
// It returns ReleaseGraph as ReleaseGraphFunc.
func BackwardMany(xs ...Node) ReleaseGraphFunc {
	ops := filterOperators(xs)
	if len(ops) == 0 {
		return nil
	}

	prepareOperatorsForBackward(ops)
	initOutputGradsForOperators(ops)
	executeBackwardPass(ops)

	return func() {
		ReleaseGraph(xs...)
	}
}

// validateGradInput checks that the input gradients are valid.
func validateGradInput(grad []mat.Matrix) {
	if len(grad) > 1 {
		panic("ag: only none or one gradients matrix must be passed to Backward")
	}
}

// getOutputGrad returns the output gradients if any.
func getOutputGrad(grad []mat.Matrix) mat.Matrix {
	if len(grad) > 0 && grad[0] != nil {
		return grad[0]
	}
	return nil
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

// prepareOperatorsForBackward prepares the operators for the backward pass.
func prepareOperatorsForBackward(operators []*Operator) {
	for _, op := range operators {
		op.prepareBackwardPass()
	}
}

// initOutputGradsForOperators initializes the output gradients for the operators.
func initOutputGradsForOperators(operators []*Operator) {
	for _, op := range operators {
		op.initOutputGrad(nil) // Implement this method in Operator
	}
}

// executeBackwardPass executes the backward pass for the operators.
func executeBackwardPass(operators []*Operator) {
	wg := &sync.WaitGroup{}
	for _, op := range operators {
		op.executeBackwardPass(wg)
	}
	wg.Wait()
}
