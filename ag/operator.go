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
	n := getOperatorPool[T]().Get().(*Operator[T])
	*n = Operator[T]{
		graph:           g,
		timeStep:        g.curTimeStep,
		id:              -1, // set below, upon insertion
		function:        f,
		value:           nil,
		grad:            nil,
		requiresGrad:    anyNodeRequiresGrad(f.Operands()),
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
		n.setParentsCounts()
	}

	g.fWG.Add(1)
	go n.forward()

	return g.insert(n)
}

func (o *Operator[T]) setParentsCounts() {
	for _, operand := range o.Operands() {
		if operand.RequiresGrad() {
			if oo, ok := operand.(*Operator[T]); ok {
				atomic.AddUint64(&oo.parentsCount, 1)
				atomic.AddUint64(&oo.pendingGrads, 1)
			}
		}
	}
}

// ID returns the ID of the node in the graph.
func (o *Operator[_]) ID() int {
	return o.id
}

// Name returns the Name of the operator.
// The name is taken from the name of r.function via reflection.
func (o *Operator[_]) Name() string {
	value := reflect.ValueOf(o.function).Elem().Type().Name()
	return regexp.MustCompile(`\[.*\]`).ReplaceAllString(value, "") // remove generics
}

// Graph returns the graph this node belongs to.
func (o *Operator[T]) Graph() *Graph[T] {
	return o.graph
}

// Value returns the result of the function.
// If the value is null, it automatically forwards to all the nodes at the
// same time-step as this operator.
func (o *Operator[T]) Value() mat.Matrix[T] {
	if atomic.LoadUint32(&o.valueAtomicFlag) == 0 {
		o.valueMx.RLock()
		defer o.valueMx.RUnlock()
	}

	return o.value
}

// HasValue returns whether the value is not nil
func (o *Operator[T]) HasValue() bool {
	return o.Value() != nil
}

// ScalarValue returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (o *Operator[T]) ScalarValue() T {
	return o.Value().Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (o *Operator[T]) Grad() mat.Matrix[T] {
	if !o.requiresGrad {
		return nil
	}
	if o.graph.backwardInProgress && atomic.LoadUint64(&o.pendingGrads) > 0 {
		o.gradsMx.RLock()
		defer o.gradsMx.RUnlock()
	}
	return o.grad
}

// PropagateGrad accumulates the gradients to the node itself.
func (o *Operator[T]) PropagateGrad(grad mat.Matrix[T]) {
	if !o.requiresGrad {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()

	if grad != nil {
		if o.grad == nil {
			o.grad = o.Value().ZerosLike()
		}
		o.grad.AddInPlace(grad)
	}

	if !o.graph.backwardInProgress {
		return
	}

	pg := atomic.LoadUint64(&o.pendingGrads)
	if pg > 0 {
		pg = atomic.AddUint64(&o.pendingGrads, ^uint64(0)) // decrement
	}
	if pg == 0 {
		o.gradsMx.Unlock()
	}
}

// HasGrad returns true if there are accumulated gradients.
func (o *Operator[_]) HasGrad() bool {
	return o.requiresGrad && o.Grad() != nil
}

// RequiresGrad returns true if the node requires gradients.
func (o *Operator[_]) RequiresGrad() bool {
	return o.requiresGrad
}

// ZeroGrad clears the gradients.
func (o *Operator[_]) ZeroGrad() {
	if !o.requiresGrad {
		return
	}
	o.gradsMx.TryLock()
	if o.grad == nil {
		return
	}
	mat.ReleaseMatrix(o.grad) // release memory
	o.grad = nil
	o.pendingGrads = o.parentsCount
}

// TimeStep returns the time-step of the node.
func (o *Operator[_]) TimeStep() int {
	return o.timeStep
}

// Operands returns the operands of the operator.
func (o *Operator[T]) Operands() []Node[T] {
	return o.function.Operands()
}

func (o *Operator[T]) backward() {
	defer o.graph.bWG.Done()
	if !o.requiresGrad {
		return
	}
	grad := o.Grad()
	if grad == nil {
		for _, operand := range o.Operands() {
			if oo, ok := operand.(*Operator[T]); ok {
				oo.PropagateGrad(nil)
			}
		}
		return
	}
	o.function.Backward(grad)
}

func (o *Operator[T]) forward() {
	o.value = o.function.Forward()
	atomic.StoreUint32(&o.valueAtomicFlag, 1)
	o.valueMx.Unlock()
	o.graph.fWG.Done()
}

func (o *Operator[_]) setID(id int) {
	o.id = id
}

// releaseValue sets the operator's value to nil releases the memory.
func (o *Operator[_]) releaseValue() {
	if o.value == nil {
		return
	}
	mat.ReleaseMatrix(o.value)
	o.value = nil
	o.valueAtomicFlag = 1
}

// releaseGrad sets the operator's gradient to nil and releases the memory.
func (o *Operator[_]) releaseGrad() {
	o.ZeroGrad()
}
