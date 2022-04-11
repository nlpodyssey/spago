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
	_ Node[float32]       = &Variable[float32]{}
)

// Variable is a type of node.
type Variable[T mat.DType] struct {
	timeStep     int
	name         string
	value        mat.Matrix[T] // store the results of a forward evaluation.
	mu           sync.Mutex    // to avoid data race during gradients accumulation
	grad         mat.Matrix[T]
	requiresGrad bool
}

// NewVariable creates and returns a new node.
func NewVariable[T mat.DType](value mat.Matrix[T], requiresGrad bool) Node[T] {
	return &Variable[T]{
		timeStep:     -1,
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
	}
}

// NewVariableWithName creates and returns a new node.
func NewVariableWithName[T mat.DType](value mat.Matrix[T], requiresGrad bool, name string) Node[T] {
	return &Variable[T]{
		timeStep:     -1,
		name:         name,
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
	}
}

// NewScalar creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func NewScalar[T mat.DType](value T) Node[T] {
	return NewVariable[T](mat.NewScalar(value), false)
}

// NewScalarWithName creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func NewScalarWithName[T mat.DType](value T, name string) Node[T] {
	return NewVariableWithName[T](mat.NewScalar(value), false, name)
}

// Constant returns a scalar Node that that doesn't require gradients.
// For the same value, a previously created Node is returned without creating a new one.
// Useful for example in the case of epsilon and number like 0.0 or 1.0.
func Constant[T mat.DType](value T) Node[T] {
	return NewVariableWithName[T](mat.NewScalar(value), false, fmt.Sprint(value))
}

// Name returns the Name of the variable (it can be empty).
// Never refer to a variable by its name and use it only for debugging purposes.
// The name is set by g.NewVariableWithName().
func (r *Variable[_]) Name() string {
	return r.name
}

// Value returns the value of the variable itself.
func (r *Variable[T]) Value() mat.Matrix[T] {
	return r.value
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

// IncTimeStep increments the value of the variable's TimeStep by one.
func (r *Variable[_]) IncTimeStep() {
	r.timeStep++
}
