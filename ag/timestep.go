// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"time"

	"github.com/nlpodyssey/spago/mat"
)

// TimeStepHandler allows handling time steps associated to the nodes of a
// computational graph, making possible to perform truncated backpropagation.
type TimeStepHandler[T mat.DType] struct {
	timeStepCreatedAt map[int]int64
	nodesTimeStep     map[Node[T]]int
}

// NewTimeStepHandler creates a new TimeStepHandler.
func NewTimeStepHandler[T mat.DType]() *TimeStepHandler[T] {
	return &TimeStepHandler[T]{
		timeStepCreatedAt: make(map[int]int64, 0),
		nodesTimeStep:     make(map[Node[T]]int, 0),
	}
}

// SetTimeStep associates a time step to the given nodes. It can be called
// only once for each unique time step value.
//
// It panics it the time step is a negative value, or if a node is already
// associated to another time step.
//
// Any operator created after this call, even if not directly linked to any of
// the nodes specified here, will be implicitly associated to the same time
// step, until the method is called again, for defining the next time
// step.
func (tsh *TimeStepHandler[T]) SetTimeStep(timeStep int, nodes ...Node[T]) {
	if timeStep < 0 {
		panic(fmt.Errorf("ag: invalid negative time step value %d", timeStep))
	}
	if _, ok := tsh.timeStepCreatedAt[timeStep]; ok {
		panic(fmt.Errorf("ag: time step %d already defined", timeStep))
	}

	tsh.timeStepCreatedAt[timeStep] = time.Now().UnixNano()

	for _, node := range nodes {
		if v, ok := tsh.nodesTimeStep[node]; ok && v != timeStep {
			panic(fmt.Errorf("ag: cannot set time step %d to node with timestep %d", timeStep, v))
		}
		tsh.nodesTimeStep[node] = timeStep
	}
}

// TimeStep returns the time step associated to the node.
//
// If a node is visited for the first time, and it is an Operator, its
// creation timestamp is compared to the creation of each time step; the result
// is resolved as the time step associated to the closest preceding timestamp,
// if any, otherwise -1. In any other case, it returns -1.
//
// Results are memoized to avoid multiple computations for the same node.
func (tsh *TimeStepHandler[T]) TimeStep(node Node[T]) int {
	if ts, ok := tsh.nodesTimeStep[node]; ok {
		return ts
	}
	ts := tsh.resolveTimeStep(node)
	tsh.nodesTimeStep[node] = ts
	return ts
}

func (tsh *TimeStepHandler[T]) resolveTimeStep(node Node[T]) int {
	op, isOp := node.(*Operator[T])
	if !isOp {
		return -1
	}
	opCreationTime := op.createdAt

	var closestCreationTime int64
	closestTimeStep := -1
	for timeStep, creationTime := range tsh.timeStepCreatedAt {
		if creationTime < opCreationTime && creationTime > closestCreationTime {
			closestCreationTime = creationTime
			closestTimeStep = timeStep
		}
	}
	return closestTimeStep
}
