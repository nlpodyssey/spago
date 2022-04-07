// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
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
	if len(grad) > 0 && grad[0] != nil && x.HasGrad() {
		panic("ag: attempt to start a backward with output gradients on a node that already has gradients.")
	}

	op, ok := x.(*Operator[T])
	if !ok {
		return
	}

	setPendingGrads(op)

	if op.grad != nil {
		op.pendingGrads--
	} else {
		var gx mat.Matrix[T]
		if len(grad) == 0 || grad[0] == nil {
			gx = x.Value().OnesLike()
			defer mat.ReleaseMatrix(gx)
		} else {
			gx = grad[0]
		}
		x.AccGrad(gx)
	}

	backward(op)
	x.Graph().bWG.Wait()
}

// BackwardMany performs the backpropagation from a list of nodes.
// This is particularly useful when there are previously assigned
// gradients on root nodes or when there are multiple distinct losses.
func BackwardMany[T mat.DType](xs ...Node[T]) {
	for _, x := range xs {
		if op, ok := x.(*Operator[T]); ok {
			setPendingGrads(op)
		}
	}
	for _, x := range xs {
		if op, ok := x.(*Operator[T]); ok {
			if op.grad != nil {
				op.pendingGrads--
			} else {
				gx := x.Value().OnesLike()
				x.AccGrad(gx)
				mat.ReleaseMatrix(gx)
			}
		}
	}
	for _, x := range xs {
		if op, ok := x.(*Operator[T]); ok {
			backward(op)
		}
	}
	xs[0].Graph().bWG.Wait()
}

func setPendingGrads[T mat.DType](op *Operator[T]) {
	if !op.requiresGrad {
		return
	}

	if op.pendingGrads < 0 {
		op.pendingGrads = 0
	}
	op.pendingGrads++
	op.gradMx.TryLock()

	if op.visited == 1 {
		return
	}
	op.visited = 1
	op.inBackward = true

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			setPendingGrads(oo)
		}
	}
}

func backward[T mat.DType](op *Operator[T]) {
	if !op.requiresGrad {
		return
	}
	if op.visited == 0 {
		return
	}
	op.visited = 0

	op.Graph().bWG.Add(1)
	go op.backward()

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			backward(oo)
		}
	}
}
