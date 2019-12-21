// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"brillion.io/spago/pkg/mat"
	"brillion.io/spago/pkg/ml/ag/fn"
	"sync"
)

type Operator struct {
	graph        *Graph
	id           int64
	function     fn.Function
	value        mat.Matrix // store the results of a forward evaluation
	mu           sync.Mutex // to avoid data race during gradients accumulation
	grad         mat.Matrix
	hasGrad      bool
	requiresGrad bool
}

// Id returns the id of the node in the graph.
func (r *Operator) Id() int64 {
	return r.id
}

// Graph returns the graph this node belongs to.
func (r *Operator) Graph() *Graph {
	return r.graph
}

// Value returns the cached result of the function.
func (r *Operator) Value() mat.Matrix {
	return r.value
}

// ScalarValue() returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *Operator) ScalarValue() float64 {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Operator) Grad() mat.Matrix {
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *Operator) PropagateGrad(grad mat.Matrix) {
	if r.requiresGrad {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.grad == nil {
			r.grad = r.value.ZerosLike()
		}
		r.grad.AddInPlace(grad)
		r.hasGrad = true
	}
}

// HasGrad returns true if there are accumulated gradients.
func (r *Operator) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the node requires gradients.
func (r *Operator) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *Operator) ZeroGrad() {
	if r.hasGrad {
		r.grad.Zeros()
		r.hasGrad = false
	} else if r.grad == nil && r.requiresGrad {
		r.grad = r.value.ZerosLike()
	}
}

func (r *Operator) backward() {
	if r.HasGrad() {
		r.function.Backward(r.grad)
	}
}
