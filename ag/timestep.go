// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"sync/atomic"

	"github.com/nlpodyssey/spago/mat"
)

var tsCounter uint64

// TimeStepHandler allows handling time steps associated to the nodes of a
// computational graph, making possible to perform truncated backpropagation.
//
// The initial implicit time step is 0, then it can be incremented with
// IncTimeStep.
type TimeStepHandler struct {
	timeSteps []uint64
}

// NewTimeStepHandler creates a new TimeStepHandler.
func NewTimeStepHandler() *TimeStepHandler {
	return &TimeStepHandler{
		timeSteps: []uint64{0},
	}
}

// IncTimeStep increments the time step by 1, keeping track of when this
// operation is performed.
func (tsh *TimeStepHandler) IncTimeStep() {
	tsh.timeSteps = append(tsh.timeSteps, atomic.AddUint64(&tsCounter, 1))
}

// CurrentTimeStep returns the current time step value.
func (tsh *TimeStepHandler) CurrentTimeStep() int {
	return len(tsh.timeSteps) - 1
}

// NodeTimeStep resolve the time step of a node belonging to the computational
// graph associated to this handler.
//
// If the node is an Operator, a Variable or a Wrapper, its creation timestamp
// is compared to the creation timestamp of each time step; the result is resolved
// as the time step associated to the closest preceding timestamp, if any, otherwise 0.
// In any other case, it returns 0.
func NodeTimeStep[T mat.DType](h *TimeStepHandler, node Node[T]) int {
	var nodeCreatedAt uint64
	switch n := node.(type) {
	case *Operator[T]:
		nodeCreatedAt = n.createdAt
	case *Variable[T]:
		nodeCreatedAt = n.createdAt
	case *Wrapper[T]:
		nodeCreatedAt = n.createdAt
	default:
		return 0
	}

	timeSteps := h.timeSteps
	for i := len(timeSteps) - 1; i >= 0; i-- {
		if timeSteps[i] <= nodeCreatedAt {
			return i
		}
	}
	return 0
}
