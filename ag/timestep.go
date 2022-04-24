// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "fmt"

// TimeStepper is implemented by any value that has a TimeStep method,
// which returns the timestep associated to an entity of a computational
// graph (some implementation of Node).
// A valid timestep is any value >= 0.
// A negative value indicates that no timestep has been associated.
type TimeStepper interface {
	TimeStep() int
}

// TimeStepSetter is implemented by any value that has a SetTimeStep method,
// which sets the timestep value associated to an entity of a computational
// graph (some implementation of Node).
// A valid timestep is any value >= 0.
// The timestep can be "unset" by setting it to a negative value.
type TimeStepSetter interface {
	SetTimeStep(int)
}

// TimeStep attempts to get the timestep associated to the given value:
// if the value satisfies the TimeStepper interface, the value of
// TimeStepper.TimeStep is returned, otherwise -1.
func TimeStep(v any) int {
	if ts, ok := v.(TimeStepper); ok {
		return ts.TimeStep()
	}
	return -1
}

// SetTimeStep is a utility method that attempts to set the given timestep
// value to the given element. The operation succeeds only if the element
// implements the interface TimeStepSetter, otherwise it panics.
func SetTimeStep(to any, timeStep int) {
	tss, ok := to.(TimeStepSetter)
	if !ok {
		panic(fmt.Errorf("ag: %T does not implement TimeStepSetter", to))
	}
	tss.SetTimeStep(timeStep)
}
