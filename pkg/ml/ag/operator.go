// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"sync"
)

var (
	_ fn.Operand = &operator{}
	_ GradValue  = &operator{}
	_ Node       = &operator{}
)

type operator struct {
	graph        *Graph
	timeStep     int64
	id           int64
	function     fn.Function
	operands     []Node
	value        mat.Matrix // store the results of a forward evaluation
	mu           sync.Mutex // to avoid data race during gradients accumulation
	grad         mat.Matrix // TODO: support of sparse gradients
	hasGrad      bool
	requiresGrad bool
}

// ID returns the ID of the node in the graph.
func (r *operator) ID() int64 {
	return r.id
}

// Graph returns the graph this node belongs to.
func (r *operator) Graph() *Graph {
	return r.graph
}

// Value returns the cached result of the function.
func (r *operator) Value() mat.Matrix {
	return r.value
}

// ScalarValue() returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *operator) ScalarValue() float64 {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *operator) Grad() mat.Matrix {
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *operator) PropagateGrad(grad mat.Matrix) {
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
func (r *operator) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the node requires gradients.
func (r *operator) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *operator) ZeroGrad() {
	if r.grad == nil {
		return
	}
	defer mat.ReleaseDense(r.grad.(*mat.Dense)) // release memory
	r.grad = nil
	r.hasGrad = false
}

func (r *operator) getTimeStep() int64 {
	return r.timeStep
}

func (r *operator) backward() {
	if !r.hasGrad {
		return
	}
	r.function.Backward(r.grad)
}
