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
	_ fn.Operand = &Operator{}
	_ Node       = &Operator{}
)

// Operator is a type of node.
type Operator struct {
	requiresGrad  int8 // -1 = undefined, 0 = false, 1 = true
	backwardState backwardState
	function      fn.Function[Node]
	// value is the results of a forward evaluation, as mat.Matrix.
	value atomic.Value
	// cond is the condition variable used as rendezvous points for
	// goroutines involved in both forward and backward operations.
	// NewOperator sets cond.L to &mx.
	cond sync.Cond
	// mx is the mutex used by cond.
	// It's defined here in order to avoid an extra memory allocation
	// (from NewOperator), but it's never be used directly.
	mx           sync.Mutex
	grad         mat.Matrix
	pendingGrads int64
	// It's primarily useful for later associating a correct time-step
	// to this operator, if needed for truncated backpropagation.
	createdAt uint64
}

// NewOperator creates a new operator along with its forward pass.
func NewOperator(f fn.Function[Node]) Node {
	op := &Operator{
		requiresGrad:  -1, // lazy evaluation
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
func (o *Operator) Name() string {
	name := reflect.ValueOf(o.function).Elem().Type().Name()
	// Strip trailing generics, if any: "foo[bar]" becomes "foo".
	if i := strings.IndexByte(name, '['); i != -1 {
		return name[:i]
	}
	return name
}

// Operands returns the operands of the operator.
func (o *Operator) Operands() []Node {
	return o.function.Operands()
}

// Value returns the result of the function.
func (o *Operator) Value() mat.Matrix {
	if v := o.value.Load(); v != nil {
		return v.(mat.Matrix)
	}

	o.cond.L.Lock()
	defer o.cond.L.Unlock()
	for {
		if v := o.value.Load(); v != nil {
			return v.(mat.Matrix)
		}
		o.cond.Wait()
	}
}

// Grad returns the gradients accumulated during the backward pass.
func (o *Operator) Grad() mat.Matrix {
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
func (o *Operator) HasGrad() bool {
	return o.Grad() != nil
}

// RequiresGrad returns true if the node requires gradients.
func (o *Operator) RequiresGrad() bool {
	if o.requiresGrad == -1 {
		o.requiresGrad = 0
		for _, op := range o.function.Operands() {
			if op.RequiresGrad() {
				o.requiresGrad = 1
				return true
			}
		}
	}
	return o.requiresGrad != 0
}

// ZeroGrad clears the gradients.
func (o *Operator) ZeroGrad() {
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
func (o *Operator) AccGrad(grad mat.Matrix) {
	if !o.RequiresGrad() {
		return
	}
	o.cond.L.Lock()
	defer o.cond.L.Unlock()

	// It is possible to observe `o.grad != nil` and at the same time `reflect.ValueOf(o.grad).IsNil() == true`.
	// That means somewhere a nil pointer is being cast to `mat.Matrix` and stored in `o.grad`.
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

func (o *Operator) initOutputGrad(outputGrad mat.Matrix) {
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
func (o *Operator) forward() {
	o.value.Store(o.function.Forward())
	o.cond.L.Lock()
	o.cond.Broadcast()
	o.cond.L.Unlock()
}

// backward executes the backward
func (o *Operator) backward() {
	if !o.RequiresGrad() {
		return
	}

	grad := o.Grad()
	if grad == nil {
		return
	}
	o.function.Backward(grad)
}

// releaseValue sets the operator's value to nil releases the memory.
func (o *Operator) releaseValue() {
	value := o.Value() // also safely waits for any forward goroutine to finish
	mat.ReleaseMatrix(value)
	o.value = atomic.Value{}
}
