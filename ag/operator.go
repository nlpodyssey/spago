// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
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

// Operator is a type of node.
type Operator[T mat.DType] struct {
	graph           *Graph[T]
	timeStep        int
	id              int
	function        fn.Function[T, Node[T]]
	value           mat.Matrix[T] // store the results of a forward evaluation
	mu              sync.Mutex    // to avoid data race during gradients accumulation TODO: rename (vs. gradsMx)
	grad            mat.Matrix[T]
	requiresGrad    bool
	valueMx         *sync.RWMutex
	valueAtomicFlag uint32
	gradAtomicFlag  uint32
	parentsCount    uint64
	pendingGrads    uint64
	gradsMx         *sync.RWMutex
}

// NewOperator creates a new operator along with its forward pass.
// Please note that operations on nodes belonging to different graphs
// result in unpredictable outcomes.
// If you are working with two or more graphs simultaneously, you may
// consider wrapping the nodes you need with NewWrap().
func (g *Graph[T]) NewOperator(f fn.Function[T, Node[T]]) Node[T] {
	operands := f.Operands()

	n := getOperatorPool[T]().Get().(*Operator[T])
	*n = Operator[T]{
		graph:           g,
		timeStep:        g.curTimeStep,
		id:              -1, // set below, upon insertion
		function:        f,
		value:           nil,
		grad:            nil,
		requiresGrad:    anyNodeRequiresGrad(operands),
		valueMx:         new(sync.RWMutex),
		valueAtomicFlag: 0,
		parentsCount:    0,
		pendingGrads:    0,
		gradsMx:         nil,
	}
	n.valueMx.Lock()
	if n.requiresGrad {
		n.gradsMx = new(sync.RWMutex)
		n.gradsMx.Lock()
		n.setParentsCounts(operands)
	}

	g.fWG.Add(1)
	go n.forward()

	return g.insert(n)
}

func (r *Operator[T]) setParentsCounts(operands []Node[T]) {
	for _, o := range operands {
		if o.RequiresGrad() {
			if oo, ok := o.(*Operator[T]); ok {
				atomic.AddUint64(&oo.parentsCount, 1)
				atomic.AddUint64(&oo.pendingGrads, 1)
			}
		}
	}
}

// ID returns the ID of the node in the graph.
func (r *Operator[_]) ID() int {
	return r.id
}

// Name returns the Name of the operator.
// The name is taken from the name of r.function via reflection.
func (r *Operator[_]) Name() string {
	value := reflect.ValueOf(r.function).Elem().Type().Name()
	return regexp.MustCompile(`\[.*\]`).ReplaceAllString(value, "") // remove generics
}

// Graph returns the graph this node belongs to.
func (r *Operator[T]) Graph() *Graph[T] {
	return r.graph
}

// Value returns the result of the function.
// If the value is null, it automatically forwards to all the nodes at the
// same time-step as this operator.
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
	return r.Value().Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (r *Operator[T]) Grad() mat.Matrix[T] {
	if !r.requiresGrad {
		return nil
	}
	if r.graph.backwardInProgress && atomic.LoadUint64(&r.pendingGrads) > 0 {
		r.gradsMx.RLock()
		defer r.gradsMx.RUnlock()
	}
	return r.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (r *Operator[T]) PropagateGrad(grad mat.Matrix[T]) {
	if !r.requiresGrad {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if grad != nil {
		if r.grad == nil {
			r.grad = r.value.ZerosLike()
		}
		r.grad.AddInPlace(grad)
	}

	if !r.graph.backwardInProgress {
		return
	}

	pg := atomic.LoadUint64(&r.pendingGrads)
	if pg > 0 {
		pg = atomic.AddUint64(&r.pendingGrads, ^uint64(0)) // decrement
	}
	if pg == 0 {
		r.gradsMx.Unlock()
	}
}

// HasGrad returns true if there are accumulated gradients.
func (r *Operator[_]) HasGrad() bool {
	return r.requiresGrad && r.Grad() != nil
}

// RequiresGrad returns true if the node requires gradients.
func (r *Operator[_]) RequiresGrad() bool {
	return r.requiresGrad
}

// ZeroGrad clears the gradients.
func (r *Operator[_]) ZeroGrad() {
	if !r.requiresGrad {
		return
	}
	r.gradsMx.TryLock()
	if r.grad == nil {
		return
	}
	mat.ReleaseMatrix(r.grad) // release memory
	r.grad = nil
	r.pendingGrads = r.parentsCount
}

// TimeStep returns the time-step of the node.
func (r *Operator[_]) TimeStep() int {
	return r.timeStep
}

// Operands returns the operands of the operator.
func (r *Operator[T]) Operands() []Node[T] {
	return r.function.Operands()
}

func (r *Operator[T]) backward() {
	defer r.graph.bWG.Done()
	if !r.requiresGrad {
		return
	}
	grad := r.Grad()
	if grad == nil {
		for _, o := range r.Operands() {
			if oo, ok := o.(*Operator[T]); ok {
				oo.PropagateGrad(nil)
			}
		}
		return
	}
	r.function.Backward(grad)
}

func (r *Operator[T]) forward() {
	defer r.graph.fWG.Done()
	r.value = r.function.Forward()
	atomic.StoreUint32(&r.valueAtomicFlag, 1)
	r.valueMx.Unlock()
}

func (r *Operator[_]) setID(id int) {
	r.id = id
}

// releaseValue sets the operator's value to nil releases the memory.
func (r *Operator[_]) releaseValue() {
	if r.value == nil {
		return
	}
	mat.ReleaseMatrix(r.value)
	r.value = nil
	r.valueAtomicFlag = 1
}

// releaseGrad sets the operator's gradient to nil and releases the memory.
func (r *Operator[_]) releaseGrad() {
	r.ZeroGrad()
	r.gradAtomicFlag = 1
}
