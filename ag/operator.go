// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"reflect"
	"strings"
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
	requiresGrad  bool
	backwardState backwardState
	function      fn.Function[T, Node[T]]
	// value is the results of a forward evaluation, as mat.Matrix[T].
	value atomic.Value
	// cond is the condition variable used as rendezvous points for
	// goroutines involved in both forward and backward operations.
	// NewOperator sets cond.L to &mx.
	cond sync.Cond
	// mx is the mutex used by cond.
	// It's defined here in order to avoid an extra memory allocation
	// (from NewOperator), but it's never be used directly.
	mx           sync.Mutex
	grad         mat.Matrix[T]
	pendingGrads int64
	// It's primarily useful for later associating a correct time-step
	// to this operator, if needed for truncated backpropagation.
	createdAt uint64
}

// NewOperator creates a new operator along with its forward pass.
// Please note that operations on nodes belonging to different graphs
// result in unpredictable outcomes.
// If you are working with two or more graphs simultaneously, you may
// consider wrapping the nodes you need with NewWrap().
func NewOperator[T mat.DType](f fn.Function[T, Node[T]]) Node[T] {
	requiresGrad := false
	for _, n := range f.Operands() {
		if !requiresGrad && n.RequiresGrad() {
			requiresGrad = true
		}
	}

	op := &Operator[T]{
		requiresGrad:  requiresGrad,
		backwardState: idle,
		function:      f,
		pendingGrads:  0,
		createdAt:     atomic.LoadUint64(&tsCounter),
	}

	op.cond.L = &op.mx

	go op.forward()

	return op
}

// Name returns the Name of the operator.
// The name is taken from the name of r.function via reflection.
func (o *Operator[_]) Name() string {
	name := reflect.ValueOf(o.function).Elem().Type().Name()
	// Strip trailing generics, if any: "foo[bar]" becomes "foo".
	if i := strings.IndexByte(name, '['); i != -1 {
		return name[:i]
	}
	return name
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

	o.cond.L.Lock()
	defer o.cond.L.Unlock()
	for {
		if v := o.value.Load(); v != nil {
			return v.(mat.Matrix[T])
		}
		o.cond.Wait()
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

	o.cond.L.Lock()
	defer o.cond.L.Unlock()
	for {
		if atomic.LoadInt64(&o.pendingGrads) == 0 {
			return o.grad
		}
		o.cond.Wait()
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
	o.backwardState = idle
}

// AccGrad accumulates the gradients to the node itself.
func (o *Operator[T]) AccGrad(grad mat.Matrix[T]) {
	if !o.requiresGrad {
		return
	}
	o.cond.L.Lock()
	defer o.cond.L.Unlock()

	// It is possible to observe `o.grad != nil` and at the same time `reflect.ValueOf(o.grad).IsNil() == true`.
	// That means somewhere a nil pointer is being cast to `mat.Matrix[T]` and stored in `o.grad`.
	// Since `mat.Matrix` is an interface, the "nil test" will return false but any method call will panic as
	// `mat.Dense` does not consider the possibility of a nil pointer value.
	// A bit of reflection seems to be an acceptable quick-fix solution but an in-depth investigation is needed here.
	if o.grad == nil || reflect.ValueOf(o.grad).IsNil() {
		o.grad = grad.Clone()
	} else {
		o.grad.AddInPlace(grad)
	}

	if o.backwardState != idle && atomic.AddInt64(&o.pendingGrads, -1) == 0 {
		o.cond.Broadcast() // notify all goroutines that have been waiting for the gradients
	}
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

// forward executes the function and inform all goroutines that have been waiting for the result.
func (o *Operator[T]) forward() {
	o.value.Store(o.function.Forward())
	o.cond.L.Lock()
	o.cond.Broadcast()
	o.cond.L.Unlock()
}

// backward executes the backward
func (o *Operator[T]) backward() {
	if !o.requiresGrad {
		return
	}

	grad := o.Grad()
	if grad == nil {
		return
	}
	o.function.Backward(grad)
}

// releaseValue sets the operator's value to nil releases the memory.
func (o *Operator[T]) releaseValue() {
	value := o.Value() // also safely waits for any forward goroutine to finish
	mat.ReleaseMatrix(value)
	o.value = atomic.Value{}
}
