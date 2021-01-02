// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"sync"
)

var (
	_ fn.Operand = &variable{}
	_ GradValue  = &variable{}
	_ Node       = &variable{}
)

type variable struct {
	graph        *Graph
	timeStep     int
	id           int
	value        mat.Matrix // store the results of a forward evaluation.
	mu           sync.Mutex // to avoid data race during gradients accumulation
	grad         mat.Matrix // TODO: support of sparse gradients
	hasGrad      bool
	requiresGrad bool
}

// ID returns the ID of the node in the graph.
func (r *variable) ID() int {
	return r.id
}

// Graph returns the graph this node belongs to.
func (r *variable) Graph() *Graph {
	return r.graph
}

// Value returns the value of the variable itself.
func (r *variable) Value() mat.Matrix {
	return r.value
}

// ScalarValue() returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *variable) ScalarValue() mat.Float {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *variable) Grad() mat.Matrix {
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *variable) PropagateGrad(grad mat.Matrix) {
	if !r.requiresGrad {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.grad == nil {
		r.grad = mat.GetEmptyDenseWorkspace(r.value.Dims()) // this could reduce the number of allocations
	}
	r.grad.AddInPlace(grad)
	r.hasGrad = true
}

// HasGrad returns true if there are accumulated gradients.
func (r *variable) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the node requires gradients.
func (r *variable) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *variable) ZeroGrad() {
	if r.grad == nil {
		return
	}
	defer mat.ReleaseDense(r.grad.(*mat.Dense)) // release memory
	r.grad = nil
	r.hasGrad = false
}

func (r *variable) TimeStep() int {
	return r.timeStep
}
