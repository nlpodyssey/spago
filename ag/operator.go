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
	_ Node[float32]       = &Operator[float32]{}
)

// Operator is a type of node.
type Operator[T mat.DType] struct {
	inBackward   bool
	requiresGrad bool
	visited      bool
	timeStep     int
	function     fn.Function[T, Node[T]]
	// value is the results of a forward evaluation, as mat.Matrix[T].
	value atomic.Value
	// valueMx is the mutex used by valueCond. It's kept here to avoid an
	// extra memory allocation, but it shouldn't be used directly.
	valueMx sync.Mutex
	// valueCond.L is set to &valueMx
	valueCond sync.Cond
	grad      mat.Matrix[T]
	// gradMx is the mutex used by gradCond. It's kept here to avoid an
	// extra memory allocation, but it shouldn't be used directly.
	gradMx sync.Mutex
	// gradCond.L is set to &gradMx
	gradCond     sync.Cond
	pendingGrads int64
}

// NewOperator creates a new operator along with its forward pass.
// Please note that operations on nodes belonging to different graphs
// result in unpredictable outcomes.
// If you are working with two or more graphs simultaneously, you may
// consider wrapping the nodes you need with NewWrap().
func NewOperator[T mat.DType](f fn.Function[T, Node[T]]) Node[T] {
	n := &Operator[T]{
		requiresGrad: anyNodeRequiresGrad(f.Operands()),
		visited:      false,
		timeStep:     -1,
		function:     f,
		pendingGrads: 0,
	}

	n.valueCond.L = &n.valueMx
	n.gradCond.L = &n.gradMx

	go n.forward()

	return n
}

// Name returns the Name of the operator.
// The name is taken from the name of r.function via reflection.
func (o *Operator[_]) Name() string {
	value := reflect.ValueOf(o.function).Elem().Type().Name()
	return regexp.MustCompile(`\[.*\]`).ReplaceAllString(value, "") // remove generics
}

// Operands returns the operands of the operator.
func (o *Operator[T]) Operands() []Node[T] {
	return o.function.Operands()
}

// Value returns the result of the function.
func (o *Operator[T]) Value() mat.Matrix[T] {
	if v := o.value.Load(); v != nil {
		return v.(mat.Matrix[T])
	}

	o.valueCond.L.Lock()
	defer o.valueCond.L.Unlock()
	for {
		if v := o.value.Load(); v != nil {
			return v.(mat.Matrix[T])
		}
		o.valueCond.Wait()
	}
}

// Grad returns the gradients accumulated during the backward pass.
func (o *Operator[T]) Grad() mat.Matrix[T] {
	if !o.requiresGrad {
		return nil
	}

	if atomic.LoadInt64(&o.pendingGrads) == 0 {
		return o.grad
	}

	o.gradCond.L.Lock()
	defer o.gradCond.L.Unlock()
	for {
		if atomic.LoadInt64(&o.pendingGrads) == 0 {
			return o.grad
		}
		o.gradCond.Wait()
	}
}

// HasGrad returns true if there are accumulated gradients.
func (o *Operator[_]) HasGrad() bool {
	return o.Grad() != nil
}

// RequiresGrad returns true if the node requires gradients.
func (o *Operator[_]) RequiresGrad() bool {
	return o.requiresGrad
}

// ZeroGrad clears the gradients.
func (o *Operator[_]) ZeroGrad() {
	o.Grad() // safety wait for the backward goroutine to finish
	if o.grad == nil {
		return
	}
	mat.ReleaseMatrix(o.grad)
	o.grad = nil
	o.pendingGrads = 0
	o.visited = false
	o.inBackward = false
}

// AccGrad accumulates the gradients to the node itself.
func (o *Operator[T]) AccGrad(grad mat.Matrix[T]) {
	if !o.requiresGrad {
		return
	}
	o.gradCond.L.Lock()
	defer o.gradCond.L.Unlock()

	if o.grad == nil {
		o.grad = o.Value().ZerosLike()
	}
	o.grad.AddInPlace(grad)

	if o.inBackward && atomic.AddInt64(&o.pendingGrads, -1) == 0 {
		o.gradCond.Broadcast() // notify all goroutines that have been waiting for the gradients
	}
}

// TimeStep returns the time-step of the node.
func (o *Operator[_]) TimeStep() int {
	return o.timeStep
}

// IncTimeStep increments the value of the operator's TimeStep by one.
func (o *Operator[_]) IncTimeStep() {
	o.timeStep++
}

func (o *Operator[T]) initOutputGrad(outputGrad mat.Matrix[T]) {
	if outputGrad != nil && o.grad != nil {
		panic("ag: attempt to set output gradients on a node that already has gradients")
	}

	if o.grad != nil {
		o.pendingGrads--
		return
	}

	if outputGrad != nil {
		o.AccGrad(outputGrad)
		return
	}

	gx := o.Value().OnesLike()
	o.AccGrad(gx)
	mat.ReleaseMatrix(gx)
}

func anyNodeRequiresGrad[T mat.DType](nodes []Node[T]) bool {
	for _, node := range nodes {
		if node.RequiresGrad() {
			return true
		}
	}
	return false
}

// forward executes the function and inform all goroutines that have been waiting for the result.
func (o *Operator[T]) forward() {
	o.value.Store(o.function.Forward())
	o.valueCond.L.Lock()
	o.valueCond.Broadcast()
	o.valueCond.L.Unlock()
}

// backward executes the backward
func (o *Operator[T]) backward() {
	defer func() {
		o.inBackward = false
	}()

	if !o.requiresGrad {
		return
	}

	grad := o.Grad() // wait for the forward goroutine to finish
	if grad == nil {
		return
	}
	o.function.Backward(grad)
}

// releaseValue sets the operator's value to nil releases the memory.
func (o *Operator[T]) releaseValue() {
	o.Value() // wait for the forward goroutine to finish
	value := o.value.Load()
	if value == nil {
		return
	}
	mat.ReleaseMatrix(value.(mat.Matrix[T]))
	o.value = atomic.Value{}
}
