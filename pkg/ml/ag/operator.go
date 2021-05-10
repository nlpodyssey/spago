// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"reflect"
	"sync"
)

var (
	_ fn.Operand = &Operator{}
	_ GradValue  = &Operator{}
	_ Node       = &Operator{}
)

var operatorPool = sync.Pool{
	New: func() interface{} {
		return new(Operator)
	},
}

// Operator is a type of node.
type Operator struct {
	graph        *Graph
	timeStep     int
	id           int
	function     fn.Function
	operands     []Node
	value        mat.Matrix // store the results of a forward evaluation
	mu           sync.Mutex // to avoid data race during gradients accumulation
	grad         mat.Matrix // TODO: support of sparse gradients
	hasGrad      bool
	requiresGrad bool
}

// ID returns the ID of the node in the graph.
func (r *Operator) ID() int {
	return r.id
}

// Name returns the Name of the operator.
// The name is taken from the name of r.function via reflection.
func (r *Operator) Name() string {
	return reflect.ValueOf(r.function).Elem().Type().Name()
}

// Graph returns the graph this node belongs to.
func (r *Operator) Graph() *Graph {
	return r.graph
}

// Value returns the cached result of the function.
func (r *Operator) Value() mat.Matrix {
	return r.value
}

// ScalarValue returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *Operator) ScalarValue() mat.Float {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Operator) Grad() mat.Matrix {
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *Operator) PropagateGrad(grad mat.Matrix) {
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
func (r *Operator) HasGrad() bool {
	return r.hasGrad
}

// RequiresGrad returns true if the node requires gradients.
func (r *Operator) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *Operator) ZeroGrad() {
	if r.grad == nil {
		return
	}
	defer mat.ReleaseMatrix(r.grad) // release memory
	r.grad = nil
	r.hasGrad = false
}

// TimeStep returns the time-step of the node.
func (r *Operator) TimeStep() int {
	return r.timeStep
}

// Operands returns the operands of the operator.
func (r *Operator) Operands() []Node {
	return r.operands
}

func (r *Operator) backward() {
	if !r.hasGrad {
		return
	}
	r.function.Backward(r.grad)
}
