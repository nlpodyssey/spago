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

var (
	// waitForward indicates if the program should wait for the forward goroutine to finish before proceeding.
	// When set to true, the operators will wait for the forward goroutine to complete.
	// This can be particularly useful for debugging.
	waitForward = false
)

// SetWaitForward enables or disables waiting for the forward goroutine to finish before proceeding.
// When enabled, the operators will wait for the forward goroutine to complete.
// This setting can be particularly useful for debugging.
func SetWaitForward(enable bool) {
	waitForward = enable
}

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

// AutoGradFunction represents a function with automatic differentiation features.
// It's used to define a new operator.
type AutoGradFunction[T DualValue] interface {
	// Forward computes the output of the function.
	Forward() (mat.Matrix, error)
	// Backward computes the backward pass given the gradient of the output.
	Backward(gy mat.Matrix) error
	// Operands returns the list of operands.
	Operands() []T
}

var _ Node = &Operator{}

// Operator is a type of node.
// It's used to represent a function with automatic differentiation features.
type Operator struct {
	// value stores the results of a forward evaluation, as mat.Matrix.
	// It's set by execute() goroutine.
	// Use the Value() method to get the actual value.
	// It also contains the accumulated gradients. Use the Grad() method to get them.
	// It is important to remember that value is a weak reference, as the matrix
	// derived from graph's operations can be freed (see ReleaseGraph).
	value mat.Matrix
	// AutoGradFunction's operands are memoized here after the first request.
	operands []DualValue
	// backwardPass is the backward function to be executed.
	fn AutoGradFunction[DualValue]
	// broadcast is the channel used to broadcast the result of the forward pass.
	broadcast chan struct{}
	// broadcastGrad is the channel used to broadcast the result of the backward pass.
	// It is initialized only when the backward pass is performed.
	broadcastGrad chan struct{}
	// pendingGrads is the number of pending gradients to be accumulated. (default: 0)
	pendingGrads int64
	// requiresGrad is a flag that indicates whether the operator requires gradients.
	// It's set to -1 (undefined) by default, and it's lazily evaluated.
	// Use the RequiresGrad() method to get the actual value.
	requiresGrad int8 // -1 = undefined, 0 = false, 1 = true
	// backwardState is the state of the backward pass.
	backwardState backwardState
}

// NewOperator creates a new operator performing the given function in a separate goroutine.
func NewOperator(f AutoGradFunction[DualValue]) DualValue {
	op := &Operator{
		requiresGrad:  -1,
		backwardState: idle,
		fn:            f,
		broadcast:     make(chan struct{}, 0),
	}

	go op.execute()

	if waitForward {
		op.Value() // wait for the forward goroutine to finish
	}
	return op
}

// forward executes the forward function and inform all goroutines that have been waiting for the result.
func (o *Operator) execute() {
	var err error
	if o.value, err = o.fn.Forward(); err != nil {
		log.Fatalf("ag: error during forward pass: %v", err) // TODO: handle error
	}
	close(o.broadcast) // inform all goroutines that have been waiting for the result
}

// Value returns the result of the function.
func (o *Operator) Value() mat.Matrix {
	<-o.broadcast // wait for the forward goroutine to finish
	return o.value
}

// Grad returns the gradients accumulated during the backward pass.
func (o *Operator) Grad() mat.Matrix {
	if o.backwardState == idle || atomic.LoadInt64(&o.pendingGrads) == 0 {
		return o.Value().Grad()
	}

	<-o.broadcastGrad // wait for the backward goroutine to finish
	return o.Value().Grad()
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
func (o *Operator) Operands() []DualValue {
	if o.operands == nil {
		o.operands = o.fn.Operands()
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
	o.Value().AccGrad(grad)

	// Don't decrement the counter if the backward pass is not running.
	if o.backwardState != idle && atomic.AddInt64(&o.pendingGrads, -1) == 0 {
		close(o.broadcastGrad) // notify all goroutines that have been waiting for the gradients
	}
}

// isNil returns true if the gradients are nil.
func isNil(grad mat.Matrix) bool {
	if grad == nil || reflect.ValueOf(grad).IsNil() {
		return true
	}
	return false
}

func (o *Operator) setOutputGrad() {
	if isNil(o.Value().Grad()) {
		gx := o.Value().OnesLike()
		o.AccGrad(gx)
		mat.ReleaseMatrix(gx)
		return
	}
	// If the node already has gradients, we can use them directly.
	o.pendingGrads--
}

func (o *Operator) prepareBackwardPass() {
	if !o.RequiresGrad() {
		return
	}
	o.pendingGrads++
	if o.backwardState == idle {
		o.backwardState = pending
		o.broadcastGrad = make(chan struct{}, 0)
		o.traverseOperandsForPreparation()
	}
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
		if err := o.fn.Backward(grad); err != nil {
			log.Fatalf("ag: error during backward pass: %v", err) // TODO: handle error
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
	o.value = nil
}
