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
	gradMx    sync.RWMutex
	// gradAccMx prevents race conditions during gradients accumulation.
	gradAccMx    sync.Mutex
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

	if n.requiresGrad {
		n.gradMx.Lock()
	}

	go n.forward()

	return n
}

func anyNodeRequiresGrad[T mat.DType](nodes []Node[T]) bool {
	for _, node := range nodes {
		if node.RequiresGrad() {
			return true
		}
	}
	return false
}

// Name returns the Name of the operator.
// The name is taken from the name of r.function via reflection.
func (o *Operator[_]) Name() string {
	value := reflect.ValueOf(o.function).Elem().Type().Name()
	return regexp.MustCompile(`\[.*\]`).ReplaceAllString(value, "") // remove generics
}

// TimeStep returns the time-step of the node.
func (o *Operator[_]) TimeStep() int {
	return o.timeStep
}

// IncTimeStep increments the value of the operator's TimeStep by one.
func (o *Operator[_]) IncTimeStep() {
	o.timeStep++
}

// Operands returns the operands of the operator.
func (o *Operator[T]) Operands() []Node[T] {
	return o.function.Operands()
}

func (o *Operator[T]) forward() {
	o.value.Store(o.function.Forward())
	o.valueCond.L.Lock()
	o.valueCond.Broadcast()
	o.valueCond.L.Unlock()
}

func (o *Operator[T]) backward() {
	defer func() {
		o.inBackward = false
	}()

	if !o.requiresGrad {
		return
	}

	grad := o.Grad()
	if grad == nil {
		return
	}
	o.function.Backward(grad)
}

// ReleaseGraph traverses the (sub-)graphs consisting of operators and
// nested operands, starting from the given nodes, and frees the resources
// of each operator.
//
// Any Node implementation can be passed to the function, however only Operators
// and their operands will be taken into account, and the rest simply ignored.
//
// This function is not concurrency safe.
//
// Freed resources include, but are not limited to, the value and the gradients.
// Any freed operator MUST not be used after this operation is performed.
func ReleaseGraph[T mat.DType](nodes ...Node[T]) {
	visited := make(map[*Operator[T]]struct{})
	for _, node := range nodes {
		if op, ok := node.(*Operator[T]); ok {
			releaseGraph[T](visited, op)
		}
	}
}

func releaseGraph[T mat.DType](visited map[*Operator[T]]struct{}, op *Operator[T]) {
	if _, ok := visited[op]; ok {
		return
	}
	visited[op] = struct{}{}

	op.releaseValue()
	op.ZeroGrad()

	for _, operand := range op.function.Operands() {
		if oo, ok := operand.(*Operator[T]); ok {
			releaseGraph[T](visited, oo)
		}
	}

	op.function = nil
	op.valueCond.L = nil
}
