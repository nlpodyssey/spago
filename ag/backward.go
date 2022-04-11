// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"sync"
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
	if len(grad) > 1 {
		panic("ag: only none or one gradients matrix must be passed to Backward")
	}

	op, ok := x.(*Operator[T])
	if !ok {
		return
	}

	setupOperatorForBackward(op)

	var outputGrad mat.Matrix[T] = nil
	if len(grad) > 0 && grad[0] != nil {
		outputGrad = grad[0]
	}
	op.initOutputGrad(outputGrad)

	wg := new(sync.WaitGroup)
	backward(wg, op)
	wg.Wait()
}

// BackwardMany performs the backpropagation from a list of nodes.
// This is particularly useful when there are previously assigned
// gradients on root nodes or when there are multiple distinct losses.
func BackwardMany[T mat.DType](xs ...Node[T]) {
	ops := make([]*Operator[T], 0, len(xs))
	for _, x := range xs {
		if op, ok := x.(*Operator[T]); ok {
			ops = append(ops, op)
		}
	}

	for _, op := range ops {
		setupOperatorForBackward(op)
	}

	for _, op := range ops {
		op.initOutputGrad(nil)
	}

	wg := new(sync.WaitGroup)
	for _, op := range ops {
		backward(wg, op)
	}
	wg.Wait()
}

func setupOperatorForBackward[T mat.DType](op *Operator[T]) {
	if !op.requiresGrad {
		return
	}

	op.pendingGrads++

	if op.visited {
		return
	}
	op.visited = true
	op.inBackward = true
	op.gradMx.TryLock()

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			setupOperatorForBackward(oo)
		}
	}
}

func backward[T mat.DType](wg *sync.WaitGroup, op *Operator[T]) {
	if !op.requiresGrad {
		return
	}
	if !op.visited {
		return
	}
	op.visited = false

	wg.Add(1)
	go func() {
		op.backward()
		wg.Done()
	}()

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			backward(wg, oo)
		}
	}
}
