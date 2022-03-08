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
	graph    *Graph[T]
	timeStep int
	id       int
	wrapGrad bool
}

// ID returns the ID of the node in the graph.
func (r *Wrapper[_]) ID() int {
	return r.id
}

// Graph returns the graph this node belongs to.
func (r *Wrapper[T]) Graph() *Graph[T] {
	return r.graph
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Wrapper[T]) Grad() mat.Matrix[T] {
	if !r.wrapGrad {
		return nil
	}
	return r.GradValue.Grad()
}

// PropagateGrad propagates the gradients to the node.
func (r *Wrapper[T]) PropagateGrad(gx mat.Matrix[T]) {
	if !r.wrapGrad {
		return
	}
	r.GradValue.PropagateGrad(gx)
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

func (r *Wrapper[_]) setID(id int) {
	r.id = id
}
