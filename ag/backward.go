// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// backwardState is an enumeration type associated to an Operator, to keep
// track of its visited status among different backpropagation phases.
type backwardState byte

const (
	// idle reports that gradient propagation is not pending for an
	// operator node.
	//
	// It's the default zero-value state of an operator, and it's also the
	// final value set from the backward step once gradients have been
	// propagated.
	//
	// As soon as a backward operation is performed, the status will change to
	// pending.
	idle backwardState = iota
	// pending is set on an operator node from the preparatory phase
	// of the backward step.
	// It reports that the node has been marked as a candidate for gradients
	// propagation and the number of pendingGrads has been computed.
	//
	// The next logical state is ongoing.
	pending
	// ongoing is set on an operator node from the core phase of the
	// backward step. It reports that the node has been visited once for
	// performing its Operator.backward method.
	//
	// This status remains set until the gradients of all dependents have been
	// resolved, and the node's own gradients have been propagated too.
	// After that, the status is set back to idle.
	ongoing
)

// Backward starts the back-propagation from the node.
//
// It follows these mutually exclusive rules:
//   a) the node has gradients already (probably assigned externally via node.AccGrads()), use those;
//   b) the output gradients are passed, use those;
//   c) the output gradients are automatically assigned by finding the derivative of the node with respect
//      to the node itself (dy/dy = 1).
//
// The gradients, except those of the given node, are summed to the existing ones during the process.
// Unless that's what you want, make sure all nodes have zero gradients.
//
// It panics if gradients are passed but the node already has them assigned.
func Backward[T mat.DType](x Node[T], grad ...mat.Matrix[T]) {
	BackwardT[T](nil, -1, x, grad...)
}

// BackwardMany performs the backpropagation from a list of nodes.
// This is particularly useful when there are previously assigned
// gradients on root nodes or when there are multiple distinct losses.
func BackwardMany[T mat.DType](xs ...Node[T]) {
	BackwardManyT[T](nil, -1, xs...)
}

// BackwardT starts a truncated backpropagation from the node.
// It works like Backward, but allows to specify a maximum number
// of backward steps compared against the nodes' time step.
//
// If backSteps is a negative value, no truncation is applied.
func BackwardT[T mat.DType](tsh *TimeStepHandler, backSteps int, x Node[T], grad ...mat.Matrix[T]) {
	if len(grad) > 1 {
		panic("ag: only none or one gradients matrix must be passed to Backward")
	}

	op, ok := x.(*Operator[T])
	if !ok {
		return
	}

	stopAtTimeStep := -1
	if tsh != nil {
		stopAtTimeStep = tsh.CurrentTimeStep() - backSteps
	}

	setupOperatorForBackward(tsh, op, stopAtTimeStep)

	var outputGrad mat.Matrix[T] = nil
	if len(grad) > 0 && grad[0] != nil {
		outputGrad = grad[0]
	}
	op.initOutputGrad(outputGrad)

	wg := new(sync.WaitGroup)
	backward(tsh, wg, op, stopAtTimeStep)
	wg.Wait()
}

// BackwardManyT starts a truncated backpropagation from a list of nodes.
// It works like BackwardMany, but allows to specify a maximum
// number of backward steps compared against the nodes' time step.
//
// If backSteps is a negative value, no truncation is applied.
func BackwardManyT[T mat.DType](tsh *TimeStepHandler, backSteps int, xs ...Node[T]) {
	ops := make([]*Operator[T], 0, len(xs))
	for _, x := range xs {
		if op, ok := x.(*Operator[T]); ok {
			ops = append(ops, op)
		}
	}

	stopAtTimeStep := -1
	if tsh != nil {
		stopAtTimeStep = tsh.CurrentTimeStep() - backSteps
	}

	for _, op := range ops {
		setupOperatorForBackward(tsh, op, stopAtTimeStep)
	}

	for _, op := range ops {
		op.initOutputGrad(nil)
	}

	wg := new(sync.WaitGroup)
	for _, op := range ops {
		backward(tsh, wg, op, stopAtTimeStep)
	}
	wg.Wait()
}

func setupOperatorForBackward[T mat.DType](tsh *TimeStepHandler, op *Operator[T], stopAtTimeStep int) {
	if !op.requiresGrad || timeStepTruncation(tsh, op, stopAtTimeStep) {
		return
	}

	op.pendingGrads++

	if op.backwardState != idle {
		return
	}
	op.backwardState = pending

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			setupOperatorForBackward(tsh, oo, stopAtTimeStep)
		}
	}
}

func backward[T mat.DType](tsh *TimeStepHandler, wg *sync.WaitGroup, op *Operator[T], stopAtTimeStep int) {
	if !op.requiresGrad || op.backwardState != pending || timeStepTruncation(tsh, op, stopAtTimeStep) {
		return
	}
	op.backwardState = ongoing

	wg.Add(1)
	go func() {
		op.backward()
		op.backwardState = idle
		wg.Done()
	}()

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			backward(tsh, wg, oo, stopAtTimeStep)
		}
	}
}

func timeStepTruncation[T mat.DType](tsh *TimeStepHandler, op *Operator[T], stopAtTimeStep int) bool {
	return tsh != nil &&
		stopAtTimeStep >= 0 &&
		tsh.NodeTimeStep(op) <= stopAtTimeStep
}
