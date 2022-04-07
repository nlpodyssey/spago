// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
	"reflect"
	"regexp"
	"sync"
	"sync/atomic"
)

var (
	_ fn.Operand[float32] = &Operator[float32]{}
	_ GradValue[float32]  = &Operator[float32]{}
	_ Node[float32]       = &Operator[float32]{}
)

// Operator is a type of node.
type Operator[T mat.DType] struct {
	graph        *Graph[T]
	timeStep     int
	id           int
	function     fn.Function[T, Node[T]]
	value        atomic.Value // store the results of a forward evaluation
	valueMx      *sync.RWMutex
	valueCond    *sync.Cond
	requiresGrad bool
	grad         mat.Matrix[T]
	gradMx       *sync.RWMutex
	gradAccMx    sync.Mutex // to avoid data race during gradients accumulation
	parentsCount int64
	pendingGrads int64
}

// NewOperator creates a new operator along with its forward pass.
// Please note that operations on nodes belonging to different graphs
// result in unpredictable outcomes.
// If you are working with two or more graphs simultaneously, you may
// consider wrapping the nodes you need with NewWrap().
func (g *Graph[T]) NewOperator(f fn.Function[T, Node[T]]) Node[T] {
	n := getOperatorPool[T]().Get().(*Operator[T])
	valueMx := new(sync.RWMutex)

	*n = Operator[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           -1, // set below, upon insertion
		function:     f,
		value:        atomic.Value{},
		valueMx:      valueMx,
		valueCond:    sync.NewCond(valueMx.RLocker()),
		requiresGrad: anyNodeRequiresGrad(f.Operands()),
		grad:         nil,
		gradMx:       nil,
		gradAccMx:    sync.Mutex{},
		parentsCount: 0,
		pendingGrads: 0,
	}

	if n.requiresGrad {
		n.gradMx = new(sync.RWMutex)
		n.gradMx.Lock()
		n.setParentsCounts()
	}

	g.fWG.Add(1)
	go n.forward()

	return g.insert(n)
}

func anyNodeRequiresGrad[T mat.DType](nodes []Node[T]) bool {
	for _, node := range nodes {
		if node.RequiresGrad() {
			return true
		}
	}
	return false
}

// ID returns the ID of the node in the graph.
func (o *Operator[_]) ID() int {
	return o.id
}

func (o *Operator[_]) setID(id int) {
	o.id = id
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

// TimeStep returns the time-step of the node.
func (o *Operator[_]) TimeStep() int {
	return o.timeStep
}

// Operands returns the operands of the operator.
func (o *Operator[T]) Operands() []Node[T] {
	return o.function.Operands()
}

func (o *Operator[T]) forward() {
	defer o.graph.fWG.Done()
	o.value.Store(o.function.Forward())
	o.valueMx.Lock()
	o.valueCond.Broadcast()
	o.valueMx.Unlock()
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

func (o *Operator[T]) setParentsCounts() {
	for _, operand := range o.Operands() {
		if operand.RequiresGrad() {
			if oo, ok := operand.(*Operator[T]); ok {
				atomic.AddInt64(&oo.parentsCount, 1)
				atomic.AddInt64(&oo.pendingGrads, 1)
			}
		}
	}
}
