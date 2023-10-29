// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"log"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

var (
	// forceSyncExecution, when set to true, forces operators to run synchronously, overriding any "async" flag in the Run() function.
	// This can be particularly useful for debugging.
	forceSyncExecution = false
)

// SetForceSyncExecution enables or disables the forcing of synchronous execution for all operators.
// When enabled, the operators will run synchronously, regardless of the "async" flag in the Run() function.
// This setting can be particularly useful for debugging.
func SetForceSyncExecution(enable bool) {
	forceSyncExecution = enable
}

// backwardState is an enumeration type associated to an Operator, to keep
// track of its visited status among different backpropagation phases.
type backwardState = uint32

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
type AutoGradFunction interface {
	// Forward computes the output of the function.
	Forward() (mat.Tensor, error)
	// Backward computes the backward pass given the gradient of the output.
	Backward(gy mat.Tensor) error
	// Operands returns the list of operands.
	Operands() []mat.Tensor
}

// forwardGuard is a buffered channel that acts as a semaphore to limit the concurrency
// of async forward operations in the Run function. Its buffer size determines the maximum
// number of forward operations that can run concurrently. Acquiring and releasing slots
// in the semaphore ensures that the concurrency level stays within the desired limit.
var forwardGuard chan struct{}

// Using runtime.NumCPU() * 2 is a common heuristic for setting the number of concurrent goroutines or the concurrency level in a Go program.
var concurrencyLimit = runtime.NumCPU() * 2

func init() {
	forwardGuard = make(chan struct{}, concurrencyLimit)
}

// Operator is a type of node.
// It's used to represent a function with automatic differentiation features.
type Operator struct {
	// value stores the results of a forward evaluation, as mat.Matrix.
	// It's set by executeForward() goroutine.
	// Use the Value() method to get the actual value.
	// It also contains the accumulated gradients. Use the Grad() method to get them.
	value mat.Tensor
	// onceOperands is used to initialize the operands only once.
	onceOperands sync.Once
	// AutoGradFunction's operands are memoized here after the first request.
	operands []mat.Tensor
	// backwardPass is the backward function to be executed.
	fn AutoGradFunction
	// broadcast is the channel used to broadcast the result of the forward pass.
	broadcast chan struct{}
	// broadcastGrad is the channel used to broadcast the result of the backward pass.
	// It is initialized only when the backward pass is performed.
	broadcastGrad chan struct{}
	// pendingGrads is the number of pending gradients to be accumulated. (default: 0)
	pendingGrads int64
	// onceRequiresGrad is used to initialize the requiresGrad only once.
	onceRequiresGrad sync.Once
	// requiresGrad is a flag that indicates whether the operator requires gradients.
	// Use the RequiresGrad() method to get the actual value.
	requiresGrad bool
	// backwardState is the state of the backward pass.
	backwardState backwardState
}

// NewOperator creates a new operator with the given AutoGradFunction.
// Note that the operator's Value() can only be accessed after calling the Run() function.
func NewOperator(f AutoGradFunction) *Operator {
	return &Operator{fn: f}
}

// SetAt sets the value at the given indices.
// It panics if the given indices are out of range.
func (o *Operator) SetAt(m mat.Tensor, indices ...int) {
	o.Value().SetAt(m, indices...)
}

// At returns the value at the given indices.
// It panics if the given indices are out of range.
func (o *Operator) At(indices ...int) mat.Tensor {
	return o.Value().At(indices...)
}

// Run starts the execution of the operator, performing the forward pass.
// If the optional async argument is set to true, the forward pass will be executed in a separate goroutine.
// The function returns a pointer to the Operator, allowing for method chaining.
func (o *Operator) Run(async ...bool) *Operator {
	isAsync := !forceSyncExecution && len(async) > 0 && async[0]

	if isAsync {
		//lint:ignore S1019 explicitly set the buffer size to 0 as the channel is used as a signal
		o.broadcast = make(chan struct{}, 0)
		forwardGuard <- struct{}{}
		go func() {
			o.executeForward()
			<-forwardGuard
		}()
		return o
	}

	o.executeForward()
	return o
}

// forward executes the forward function and inform all goroutines that have been waiting for the result.
func (o *Operator) executeForward() {
	value, err := o.fn.Forward()
	if err != nil {
		log.Fatalf("ag: error during forward pass: %v", err) // TODO: handle error
	}
	o.value = value

	if o.broadcast != nil { // if nil, it means that the operator is not async
		close(o.broadcast) // inform all goroutines that have been waiting for the result
	}
}

// Value returns the result of the function.
func (o *Operator) Value() mat.Tensor {
	if o.broadcast != nil { // if nil, it means that the operator is not async
		<-o.broadcast // wait for the forward goroutine to finish
	}
	return o.value
}

func (o *Operator) Item() float.Float {
	return o.Value().Item()
}

// Grad returns the gradients accumulated during the backward pass.
func (o *Operator) Grad() mat.Tensor {
	if o.isBackwardIdle() || atomic.LoadInt64(&o.pendingGrads) == 0 {
		return o.Value().Grad()
	}

	<-o.broadcastGrad // wait for the backward goroutine to finish
	return o.Value().Grad()
}

// HasGrad returns true if there are accumulated gradients.
func (o *Operator) HasGrad() bool {
	return !isNil(o.Grad()) // safety wait for the backward goroutine to finish
}

// RequiresGrad returns true if the node requires gradients.
func (o *Operator) RequiresGrad() bool {
	o.onceRequiresGrad.Do(func() {
		for _, op := range o.Operands() {
			if op.RequiresGrad() {
				o.requiresGrad = true // memoize the result
				o.Value().(mat.Matrix).SetRequiresGrad(true)
				return
			}
		}
	})
	return o.requiresGrad
}

// Operands returns the operands of the operator.
func (o *Operator) Operands() []mat.Tensor {
	o.onceOperands.Do(func() {
		o.operands = o.fn.Operands() // memoize the result
	})
	return o.operands
}

// ZeroGrad clears the gradients.
func (o *Operator) ZeroGrad() {
	if o.HasGrad() {
		o.Value().ZeroGrad()
	}
}

// AccGrad accumulates the gradients to the node itself.
func (o *Operator) AccGrad(grad mat.Tensor) {
	o.Value().AccGrad(grad)

	// Don't decrement the counter if the backward pass is not running.
	if !o.isBackwardIdle() && atomic.AddInt64(&o.pendingGrads, -1) == 0 {
		close(o.broadcastGrad) // notify all goroutines that have been waiting for the gradients
	}
}

func (o *Operator) assignOutputGradient() error {
	grad := o.Value().Grad()

	if !isNil(grad) {
		o.pendingGrads--
		return nil
	}

	if o.Value().Size() == 1 {
		o.AccGrad(o.Value().(mat.Matrix).NewScalar(1.))
		return nil
	}

	return fmt.Errorf("ag: missing gradient for %v", o)
}

func (o *Operator) prepareBackwardPass() {
	if !o.RequiresGrad() {
		return
	}

	o.pendingGrads++
	if !o.trySetBackwardPending() {
		return
	}

	//lint:ignore S1019 explicitly set the buffer size to 0 as the channel is used as a signal
	o.broadcastGrad = make(chan struct{}, 0)

	for _, operand := range o.Operands() {
		if oo, ok := operand.(*Operator); ok {
			oo.prepareBackwardPass()
		}
	}
}

func (o *Operator) processBackwardPass(wg *sync.WaitGroup) {
	if !o.RequiresGrad() || !o.trySetBackwardOngoing() {
		return
	}

	wg.Add(1) // decrement when the backward pass is done
	go o.executeBackward(wg)

	for _, operand := range o.Operands() {
		if oo, ok := operand.(*Operator); ok {
			oo.processBackwardPass(wg)
		}
	}
}

func (o *Operator) executeBackward(wg *sync.WaitGroup) {
	defer wg.Done()
	defer o.setBackwardIdle()

	grad := o.Grad() // wait until the accumulated gradients are ready
	if grad == nil {
		return // no gradients to propagate
	}

	if err := o.fn.Backward(grad); err != nil {
		log.Fatalf("ag: error during backward pass: %v", err) // TODO: handle error
	}
}

func (o *Operator) isBackwardIdle() bool {
	return atomic.LoadUint32(&o.backwardState) == idle
}

func (o *Operator) setBackwardIdle() {
	atomic.StoreUint32(&o.backwardState, idle)
}

func (o *Operator) trySetBackwardPending() bool {
	return atomic.CompareAndSwapUint32(&o.backwardState, idle, pending)
}

func (o *Operator) trySetBackwardOngoing() bool {
	return atomic.CompareAndSwapUint32(&o.backwardState, pending, ongoing)
}

// isNil returns true if the gradients are nil.
func isNil(grad any) bool {
	if grad == nil || reflect.ValueOf(grad).IsNil() {
		return true
	}
	return false
}
