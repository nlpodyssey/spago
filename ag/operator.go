// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"log"
	"reflect"
	"sync"
	"sync/atomic"

	"github.com/nlpodyssey/spago/mat"
)

// backwardState is an enumeration type associated to an Operator, to keep
// track of its visited status among different backpropagation phases.
type backwardState byte

const (
	// idle reports that gradient propagation is not pending for an
	// operator node.
	//
	// It's the default zero-value state of an operator, and it's also the
	// final value set from the backward step once gradients have been
	// propagated.
	//
	// As soon as a backward operation is performed, the status will change to
	// pending.
	idle backwardState = iota
	// pending is set on an operator node from the preparatory phase
	// of the backward step.
	// It reports that the node has been marked as a candidate for gradients
	// propagation and the number of pendingGrads has been computed.
	//
	// The next logical state is ongoing.
	pending
	// ongoing is set on an operator node from the core phase of the
	// backward step. It reports that the node has been visited once for
	// performing its Operator.backward method.
	//
	// This status remains set until the gradients of all dependents have been
	// resolved, and the node's own gradients have been propagated too.
	// After that, the status is set back to idle.
	ongoing
)

var _ Node = &Operator{}

// AutoGradFunction represents a function with automatic differentiation features.
// It's used to define a new operator.
type AutoGradFunction[T Node] interface {
	// Forward computes the output of the function.
	Forward() (mat.Matrix, error)
	// Backward computes the backward pass given the gradient of the output.
	Backward(gy mat.Matrix) error
	// Operands returns the list of operands.
	Operands() []T
}

// ForwardFunc is the type of the forward function.
type ForwardFunc func() (mat.Matrix, error)

// BackwardFunc is the type of the backward function.
type BackwardFunc func(gy mat.Matrix) error

// Operator is a type of node.
// It's used to represent a function with automatic differentiation features.
type Operator struct {
	// requiresGrad is a flag that indicates whether the operator requires gradients.
	// It's set to -1 (undefined) by default, and it's lazily evaluated.
	// Use the RequiresGrad() method to get the actual value.
	requiresGrad int8 // -1 = undefined, 0 = false, 1 = true
	// backwardState is the state of the backward pass.
	backwardState backwardState
	// backwardPass is the backward function to be executed.
	backwardPass BackwardFunc
	// value is the results of a forward evaluation, as mat.Matrix.
	// It's set by execute() goroutine.
	// Use the Value() method to get the actual value.
	// It also contains the accumulated gradients. Use the Grad() method to get them.
	value atomic.Value
	// cond is the condition variable used as rendezvous points for
	// goroutines involved in both forward and backward operations.
	// NewOperator sets cond.L to &mx.
	cond sync.Cond
	// mx is the mutex used by cond.
	// It's defined here in order to avoid an extra memory allocation
	// (from NewOperator), but it's never be used directly.
	mx sync.Mutex
	// pendingGrads is the number of pending gradients to be accumulated. (default: 0)
	pendingGrads int64
	// operandsFunc is the function that returns the operands of the function.
	operandsFunc func() []Node
	// AutoGradFunction's operands are memoized here after the first request.
	operands []Node
}

// NewOperator creates a new operator performing the given function in a separate goroutine.
func NewOperator(f AutoGradFunction[Node]) Node {
	op := &Operator{
		requiresGrad:  -1,
		backwardState: idle,
		backwardPass:  f.Backward,
		operandsFunc:  f.Operands,
	}
	op.cond.L = &op.mx

	go op.execute(f.Forward)

	return op
}

// forward executes the function and inform all goroutines that have been waiting for the result.
func (o *Operator) execute(f ForwardFunc) {
	y, err := f()
	if err != nil {
		log.Fatalf("ag: error during forward pass: %v", err) // TODO: handle error
	}
	o.value.Store(y) // execute the function and store the result
	o.cond.L.Lock()
	o.cond.Broadcast()
	o.cond.L.Unlock()

	if debug {
		o.Value() // wait for the forward goroutine to finish
	}
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
	if o.backwardState == idle || atomic.LoadInt64(&o.pendingGrads) == 0 {
		return o.Value().Grad()
	}

	o.cond.L.Lock()
	defer o.cond.L.Unlock()
	for {
		if atomic.LoadInt64(&o.pendingGrads) == 0 {
			return o.Value().Grad()
		}
		o.cond.Wait()
	}
}

// HasGrad returns true if there are accumulated gradients.
func (o *Operator) HasGrad() bool {
	return !isNil(o.Grad())
}

// RequiresGrad returns true if the node requires gradients.
func (o *Operator) RequiresGrad() bool {
	if o.requiresGrad == -1 {
		defer func() {
			// set the result in the underlying value
			o.Value().SetRequiresGrad(o.requiresGrad == 1)
		}()

		o.requiresGrad = 0
		for _, op := range o.Operands() {
			if op.RequiresGrad() {
				o.requiresGrad = 1
				return true
			}
		}
	}
	return o.requiresGrad != 0
}

// Operands returns the operands of the operator.
func (o *Operator) Operands() []Node {
	if o.operands == nil {
		o.operands = o.operandsFunc()
	}
	return o.operands
}

// ZeroGrad clears the gradients.
func (o *Operator) ZeroGrad() {
	if o.Grad() == nil { // safety wait for the backward goroutine to finish
		return
	}
	o.Value().ZeroGrad()
	o.pendingGrads = 0
	o.backwardState = idle
}

// AccGrad accumulates the gradients to the node itself.
func (o *Operator) AccGrad(grad mat.Matrix) {
	defer o.cond.L.Unlock()
	o.cond.L.Lock()
	o.Value().AccGrad(grad)

	// Don't decrement the counter if the backward pass is not running.
	if o.backwardState != idle && atomic.AddInt64(&o.pendingGrads, -1) == 0 {
		o.cond.Broadcast() // notify all goroutines that have been waiting for the gradients
	}
}

// isNil returns true if the gradients are nil.
func isNil(grad mat.Matrix) bool {
	if grad == nil || reflect.ValueOf(grad).IsNil() {
		return true
	}
	return false
}

func (o *Operator) initOutputGrad(outputGrad mat.Matrix) {
	if outputGrad != nil && !isNil(o.Value().Grad()) {
		panic("ag: attempt to set output gradients on a node that already has gradients")
	}

	if !isNil(o.Value().Grad()) {
		// If the node already has gradients, we can use them directly.
		o.pendingGrads--
		return
	}

	if outputGrad != nil {
		// If the output gradient is provided, we can use it directly.
		o.AccGrad(outputGrad)
		return
	}

	// If neither the node nor the output gradient is provided, we need to create a new one.
	gx := o.Value().OnesLike()
	o.AccGrad(gx)
	mat.ReleaseMatrix(gx)
}

func (o *Operator) prepareBackwardPass() {
	if !o.RequiresGrad() {
		return
	}
	o.pendingGrads++
	if o.updateBackwardState() {
		o.traverseOperandsForPreparation()
	}
}

func (o *Operator) updateBackwardState() bool {
	if o.backwardState != idle {
		return false // already in progress
	}
	o.backwardState = pending
	return true
}

func (o *Operator) traverseOperandsForPreparation() {
	for _, operand := range o.Operands() {
		if oo, ok := operand.(*Operator); ok {
			oo.prepareBackwardPass()
		}
	}
}

func (o *Operator) processBackwardPass(wg *sync.WaitGroup) {
	if !o.RequiresGrad() || o.backwardState != pending {
		return
	}
	o.backwardState = ongoing

	wg.Add(1) // decrement when the backward pass is done
	go o.executeBackward(wg)

	o.traverseOperandsForBackward(wg)
}

func (o *Operator) executeBackward(wg *sync.WaitGroup) {
	if grad := o.Grad(); grad != nil {
		err := o.backwardPass(grad)
		if err != nil {
			panic(err) // TODO: handle the error
		}
	}
	o.backwardState = idle
	wg.Done()
}

func (o *Operator) traverseOperandsForBackward(wg *sync.WaitGroup) {
	for _, operand := range o.Operands() {
		if oo, ok := operand.(*Operator); ok {
			oo.processBackwardPass(wg)
		}
	}
}

// releaseValue sets the operator's value to nil releases the memory.
func (o *Operator) releaseValue() {
	value := o.Value() // also safely waits for any forward goroutine to finish
	mat.ReleaseMatrix(value)
	o.value = atomic.Value{}
}
