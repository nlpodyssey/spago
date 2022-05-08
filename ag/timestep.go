// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"math"
	"time"
)

// TimeStepHandler allows handling time steps associated to the nodes of a
// computational graph, making possible to perform truncated backpropagation.
//
// The initial implicit time step is 0, then it can be incremented with
// IncTimeStep.
type TimeStepHandler struct {
	timeSteps []int64
}

// NewTimeStepHandler creates a new TimeStepHandler.
func NewTimeStepHandler() *TimeStepHandler {
	return &TimeStepHandler{
		timeSteps: []int64{math.MinInt64},
	}
}

// IncTimeStep increments the time step by 1, keeping track of when this
// operation is performed.
func (tsh *TimeStepHandler) IncTimeStep() {
	tsh.timeSteps = append(tsh.timeSteps, time.Now().UnixNano())
}

// CurrentTimeStep returns the current time step value.
func (tsh *TimeStepHandler) CurrentTimeStep() int {
	return len(tsh.timeSteps) - 1
}

// NodeTimeStep resolve the time step of a node belonging to the computational
// graph associated to this handler.
//
// If the node is an Operator, its creation timestamp is compared to the
// creation timestamp of each time step; the result is resolved as the time
// step associated to the closest preceding timestamp, if any, otherwise 0.
// In any other case, it returns 0.
func (tsh *TimeStepHandler) NodeTimeStep(node any) int {
	switch n := node.(type) {
	case *Operator[float32]:
		return tsh.resolveTimeStep(n.createdAt)
	case *Operator[float64]:
		return tsh.resolveTimeStep(n.createdAt)
	default:
		return 0
	}
}

func (tsh *TimeStepHandler) resolveTimeStep(nodeCreatedAt int64) int {
	timeSteps := tsh.timeSteps
	for i := len(timeSteps) - 1; i >= 0; i-- {
		if timeSteps[i] <= nodeCreatedAt {
			return i
		}
	}
	return 0
}
