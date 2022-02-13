// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
	"sync"
)

var (
	_ fn.Operand[float32] = &Variable[float32]{}
	_ GradValue[float32]  = &Variable[float32]{}
	_ Node[float32]       = &Variable[float32]{}
)

// Variable is a type of node.
type Variable[T mat.DType] struct {
	graph        *Graph[T]
	timeStep     int
	id           int
	name         string
	value        mat.Matrix[T] // store the results of a forward evaluation.
	mu           sync.Mutex    // to avoid data race during gradients accumulation
	grad         mat.Matrix[T]
	hasGrad      bool
	requiresGrad bool
}

// ID returns the ID of the node in the graph.
func (r *Variable[_]) ID() int {
	return r.id
}

// Name returns the Name of the variable (it can be empty).
// Never refer to a variable by its name and use it only for debugging purposes.
// The name is set by g.NewVariableWithName().
func (r *Variable[_]) Name() string {
	return r.name
}

// Graph returns the graph this node belongs to.
func (r *Variable[T]) Graph() *Graph[T] {
	return r.graph
}

// Value returns the value of the variable itself.
func (r *Variable[T]) Value() mat.Matrix[T] {
	return r.value
}

// ScalarValue returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *Variable[T]) ScalarValue() T {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Variable[T]) Grad() mat.Matrix[T] {
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *Variable[T]) PropagateGrad(grad mat.Matrix[T]) {
	if !r.requiresGrad {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.grad == nil {
		r.grad = r.value.ZerosLike()
	}
	r.grad.AddInPlace(grad)
	r.hasGrad = true
}

// HasGrad returns true if there are accumulated gradients.
func (r *Variable[_]) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the node requires gradients.
func (r *Variable[_]) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *Variable[_]) ZeroGrad() {
	if r.grad == nil {
		return
	}
	defer mat.ReleaseMatrix(r.grad) // release memory
	r.grad = nil
	r.hasGrad = false
}

// TimeStep returns the time-step of the node.
func (r *Variable[_]) TimeStep() int {
	return r.timeStep
}
