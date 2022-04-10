// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"sync"

	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
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
	name         string
	value        mat.Matrix[T] // store the results of a forward evaluation.
	mu           sync.Mutex    // to avoid data race during gradients accumulation
	grad         mat.Matrix[T]
	requiresGrad bool
}

// NewVariable creates and returns a new node.
func (g *Graph[T]) NewVariable(value mat.Matrix[T], requiresGrad bool) Node[T] {
	n := &Variable[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
	}
	return g.insert(n)
}

// NewVariableWithName creates and returns a new node.
func (g *Graph[T]) NewVariableWithName(value mat.Matrix[T], requiresGrad bool, name string) Node[T] {
	n := &Variable[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		name:         name,
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
	}
	return g.insert(n)
}

// NewScalar creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph[T]) NewScalar(value T) Node[T] {
	return g.NewVariable(mat.NewScalar(value), false)
}

// NewScalarWithName creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph[T]) NewScalarWithName(value T, name string) Node[T] {
	return g.NewVariableWithName(mat.NewScalar(value), false, name)
}

// Constant returns a scalar Node that that doesn't require gradients.
// For the same value, a previously created Node is returned without creating a new one.
// Useful for example in the case of epsilon and number like 0.0 or 1.0.
func (g *Graph[T]) Constant(value T) Node[T] {
	g.mu2.Lock()
	defer g.mu2.Unlock()
	if node, ok := g.constants[value]; ok {
		return node
	}
	node := g.NewVariableWithName(mat.NewScalar(value), false, fmt.Sprint(value))
	g.constants[value] = node
	return node
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

// AccGrad accumulates the gradients to the node itself.
func (r *Variable[T]) AccGrad(grad mat.Matrix[T]) {
	if !r.requiresGrad {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.grad == nil {
		r.grad = grad.Clone()
		return
	}
	r.grad.AddInPlace(grad)
}

// HasGrad returns true if there are accumulated gradients.
func (r *Variable[_]) HasGrad() bool {
	return r.grad != nil
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
}

// TimeStep returns the time-step of the node.
func (r *Variable[_]) TimeStep() int {
	return r.timeStep
}
