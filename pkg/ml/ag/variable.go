// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"saientist.dev/spago/pkg/mat"
	"sync"
)

type Variable struct {
	graph        *Graph
	id           int64
	value        mat.Matrix // store the results of a forward evaluation.
	mu           sync.Mutex // to avoid data race during gradients accumulation
	grad         mat.Matrix // lazy initialization
	hasGrad      bool
	requiresGrad bool
}

// Id returns the id of the node in the graph.
func (r *Variable) Id() int64 {
	return r.id
}

// Graph returns the graph this node belongs to.
func (r *Variable) Graph() *Graph {
	return r.graph
}

// Value returns the value of the variable itself.
func (r *Variable) Value() mat.Matrix {
	return r.value
}

func (r *Variable) ChangeValue(value mat.Matrix) {
	r.value = value
}

// ScalarValue() returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *Variable) ScalarValue() float64 {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Variable) Grad() mat.Matrix {
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *Variable) PropagateGrad(grad mat.Matrix) {
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
func (r *Variable) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the node requires gradients.
func (r *Variable) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *Variable) ZeroGrad() {
	if r.hasGrad {
		r.grad.Zeros()
		r.hasGrad = false
	} else if r.grad == nil && r.requiresGrad {
		r.grad = r.value.ZerosLike()
	}
}
