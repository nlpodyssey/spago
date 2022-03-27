// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"reflect"
	"regexp"
	"sync"
	"sync/atomic"

	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

var (
	_ fn.Operand[float32] = &Operator[float32]{}
	_ GradValue[float32]  = &Operator[float32]{}
	_ Node[float32]       = &Operator[float32]{}
)

var (
	operatorPoolFloat32 = &sync.Pool{
		New: func() any { return new(Operator[float32]) },
	}
	operatorPoolFloat64 = &sync.Pool{
		New: func() any { return new(Operator[float64]) },
	}
)

func getOperatorPool[T mat.DType]() *sync.Pool {
	// TODO: review this code once stable go 1.18 is released
	switch any(T(0)).(type) {
	case float32:
		return any(operatorPoolFloat32).(*sync.Pool)
	case float64:
		return any(operatorPoolFloat64).(*sync.Pool)
	default:
		panic(fmt.Sprintf("ag: no operator pool for type %T", T(0)))
	}
}

// Operator is a type of node.
type Operator[T mat.DType] struct {
	graph           *Graph[T]
	timeStep        int
	id              int
	function        fn.Function[T, Node[T]]
	value           mat.Matrix[T] // store the results of a forward evaluation
	mu              sync.Mutex    // to avoid data race during gradients accumulation
	grad            mat.Matrix[T]
	requiresGrad    bool
	valueMx         *sync.RWMutex
	valueAtomicFlag uint32
}

// ID returns the ID of the node in the graph.
func (r *Operator[_]) ID() int {
	return r.id
}

// Name returns the Name of the operator.
// The name is taken from the name of r.function via/ reflection.
func (r *Operator[_]) Name() string {
	value := reflect.ValueOf(r.function).Elem().Type().Name()
	return regexp.MustCompile(`\[[^]]*\]`).ReplaceAllString(value, "") // remove generics
}

// Graph returns the graph this node belongs to.
func (r *Operator[T]) Graph() *Graph[T] {
	return r.graph
}

// Value returns the result of the function.
// If execution isn't eager and the value is null, it automatically forwards
// to all the nodes at the same time-step as this operator.
func (r *Operator[T]) Value() mat.Matrix[T] {
	if r.valueMx != nil {
		r.waitForValue()
	}
	return r.value
}

func (r *Operator[T]) waitForValue() {
	if atomic.LoadUint32(&r.valueAtomicFlag) == 1 {
		return
	}

	r.valueMx.RLock()
	r.valueMx.RUnlock()
}

// HasValue returns whether the value is not nil
func (r *Operator[T]) HasValue() bool {
	return r.value != nil
}

// ScalarValue returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (r *Operator[T]) ScalarValue() T {
	return r.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Operator[T]) Grad() mat.Matrix[T] {
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *Operator[T]) PropagateGrad(grad mat.Matrix[T]) {
	if !r.requiresGrad {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.grad == nil {
		r.grad = r.value.ZerosLike()
	}
	r.grad.AddInPlace(grad)
}

// HasGrad returns true if there are accumulated gradients.
func (r *Operator[_]) HasGrad() bool {
	return r.grad != nil
}

// RequiresGrad returns true if the node requires gradients.
func (r *Operator[_]) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *Operator[_]) ZeroGrad() {
	if r.grad == nil {
		return
	}
	defer mat.ReleaseMatrix(r.grad) // release memory
	r.grad = nil
}

// TimeStep returns the time-step of the node.
func (r *Operator[_]) TimeStep() int {
	return r.timeStep
}

// Operands returns the operands of the operator.
func (r *Operator[T]) Operands() []Node[T] {
	return r.function.Operands()
}

func (r *Operator[_]) backward() {
	if r.grad == nil {
		return
	}
	r.function.Backward(r.grad)
}

func (r *Operator[_]) forward() {
	if r.value != nil {
		return
	}
	r.value = r.function.Forward()
}

func (r *Operator[T]) goForward() {
	for _, operand := range r.function.Operands() {
		if o, ok := operand.(*Operator[T]); ok {
			o.waitForValue()
		}
	}
	r.value = r.function.Forward()
	atomic.StoreUint32(&r.valueAtomicFlag, 1)
	r.valueMx.Unlock()
}

func (r *Operator[_]) setID(id int) {
	r.id = id
}
