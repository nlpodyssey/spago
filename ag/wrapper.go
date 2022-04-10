// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

var (
	_ fn.Operand[float32] = &Wrapper[float32]{}
	_ GradValue[float32]  = &Wrapper[float32]{}
	_ Node[float32]       = &Wrapper[float32]{}
)

// Wrapper is a type of node.
type Wrapper[T mat.DType] struct {
	GradValue[T]
	timeStep int
	wrapGrad bool
}

// NewWrap creates a new wrapper Node for the given value, attaching it to
// the graph.
func NewWrap[T mat.DType](value GradValue[T]) Node[T] {
	return &Wrapper[T]{
		GradValue: value,
		timeStep:  -1,
		wrapGrad:  true,
	}
}

// NewWrapNoGrad is similar to NewWrap, but it disables automatic
// differentiation on the new node.
func NewWrapNoGrad[T mat.DType](value GradValue[T]) Node[T] {
	return &Wrapper[T]{
		GradValue: value,
		timeStep:  -1,
		wrapGrad:  false,
	}
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Wrapper[T]) Grad() mat.Matrix[T] {
	if !r.wrapGrad {
		return nil
	}
	return r.GradValue.Grad()
}

// AccGrad accumulates the gradients to the node.
func (r *Wrapper[T]) AccGrad(gx mat.Matrix[T]) {
	if !r.wrapGrad {
		return
	}
	r.GradValue.AccGrad(gx)
}

// HasGrad returns true if there are accumulated gradients.
func (r *Wrapper[_]) HasGrad() bool {
	if !r.wrapGrad {
		return false
	}
	return r.GradValue.HasGrad()
}

// RequiresGrad returns true if the node requires gradients.
func (r *Wrapper[_]) RequiresGrad() bool {
	if !r.wrapGrad {
		return false
	}
	return r.GradValue.RequiresGrad()
}

// ZeroGrad set the gradients to zeros.
func (r *Wrapper[_]) ZeroGrad() {
	if !r.wrapGrad {
		return
	}
	r.GradValue.ZeroGrad()
}

// TimeStep returns the time-step of the node.
func (r *Wrapper[_]) TimeStep() int {
	return r.timeStep
}

// IncTimeStep increments the value of the wrapper's TimeStep by one.
func (r *Wrapper[_]) IncTimeStep() {
	r.timeStep++
}
