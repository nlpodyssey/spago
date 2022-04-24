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
	inBackward   bool
	requiresGrad bool
	visited      bool
	function     fn.Function[T, Node[T]]
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
		function:     f,
		pendingGrads: 0,
	}

	n.cond.L = &n.mx

	go n.forward()

	return n
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
	o.visited = false
	o.inBackward = false
}

// AccGrad accumulates the gradients to the node itself.
func (o *Operator[T]) AccGrad(grad mat.Matrix[T]) {
	if !o.requiresGrad {
		return
	}
	o.cond.L.Lock()
	defer o.cond.L.Unlock()

	if o.grad == nil {
		o.cond.L.Unlock()
		o.grad = o.Value().ZerosLike()
		o.cond.L.Lock()
	}
	o.grad.AddInPlace(grad)

	if o.inBackward && atomic.AddInt64(&o.pendingGrads, -1) == 0 {
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
	o.cond.L.Lock()
	o.cond.Broadcast()
	o.cond.L.Unlock()
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
	value := o.Value() // also safely waits for any forward goroutine to finish
	mat.ReleaseMatrix(value)
	o.value = atomic.Value{}
}
