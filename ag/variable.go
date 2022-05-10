// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

var (
	_ fn.Operand[float32] = &Variable[float32]{}
	_ Node[float32]       = &Variable[float32]{}
)

// Variable is a simple type of Node, primarily consisting of a value and
// optional gradients.
type Variable[T mat.DType] struct {
	value        mat.Matrix[T]
	grad         mat.Matrix[T]
	gradMu       sync.RWMutex
	requiresGrad bool
	name         string
	// It's primarily useful for later associating a correct time-step
	// to this variable, if needed for truncated backpropagation.
	createdAt uint64
}

// NewVariable creates a new Variable Node.
func NewVariable[T mat.DType](value mat.Matrix[T], requiresGrad bool) Node[T] {
	return &Variable[T]{
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
		createdAt:    atomic.LoadUint64(&tsCounter),
	}
}

// NewVariableWithName creates a new Variable Node with a given name.
func NewVariableWithName[T mat.DType](value mat.Matrix[T], requiresGrad bool, name string) Node[T] {
	return &Variable[T]{
		name:         name,
		value:        value,
		grad:         nil,
		requiresGrad: requiresGrad,
		createdAt:    atomic.LoadUint64(&tsCounter),
	}
}

// NewScalar creates a new scalar Variable Node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func NewScalar[T mat.DType](value T) Node[T] {
	return NewVariable[T](mat.NewScalar(value), false)
}

// NewScalarWithName creates a new scalar Variable Node that doesn't require
// gradients, with a given name
// TODO: Why shouldn't gradient be required by default?
func NewScalarWithName[T mat.DType](value T, name string) Node[T] {
	return NewVariableWithName[T](mat.NewScalar(value), false, name)
}

// Constant returns a scalar Node that that doesn't require gradients.
func Constant[T mat.DType](value T) Node[T] {
	return NewVariableWithName[T](mat.NewScalar(value), false, fmt.Sprint(value))
}

// Name returns the Name of the variable (it can be empty).
//
// Identifying a Variable solely upon its name is highly discourages.
// The name should be used solely for debugging or testing purposes.
func (r *Variable[_]) Name() string {
	return r.name
}

// Value returns the value of the variable itself.
func (r *Variable[T]) Value() mat.Matrix[T] {
	return r.value
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Variable[T]) Grad() mat.Matrix[T] {
	r.gradMu.RLock()
	defer r.gradMu.RUnlock()
	return r.grad
}

// AccGrad accumulates the gradients into the Variable.
func (r *Variable[T]) AccGrad(grad mat.Matrix[T]) {
	if !r.requiresGrad {
		return
	}
	r.gradMu.Lock()
	defer r.gradMu.Unlock()
	if r.grad == nil {
		r.grad = grad.Clone()
		return
	}
	r.grad.AddInPlace(grad)
}

// HasGrad reports whether there are accumulated gradients.
func (r *Variable[_]) HasGrad() bool {
	return r.Grad() != nil
}

// RequiresGrad reports whether the Variable requires gradients.
func (r *Variable[_]) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad zeroes the gradients, setting the value of Grad to nil.
func (r *Variable[_]) ZeroGrad() {
	if r.Grad() == nil {
		return
	}
	r.gradMu.Lock()
	defer r.gradMu.Unlock()
	mat.ReleaseMatrix(r.grad)
	r.grad = nil
}
