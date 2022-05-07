// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestTimeStepHandler(t *testing.T) {
	t.Run("float32", testTimeStepHandler[float32])
	t.Run("float64", testTimeStepHandler[float64])
}

func testTimeStepHandler[T mat.DType](t *testing.T) {
	tsh := NewTimeStepHandler[T]()

	// Simulate some parameters, with no associated time step (default -1)
	paramA := NewVariableWithName[T](mat.NewScalar[T](1), true, "Param 0")
	paramB := NewVariableWithName[T](mat.NewScalar[T](2), true, "Param 1")

	// Perform an operation still without considering time steps (again -1)
	paramsSum := Sum(paramA, paramB)

	// Time step 0
	in0 := NewVariableWithName[T](mat.NewScalar[T](3), true, "Input 0")
	tsh.SetTimeStep(0, in0)

	a0 := Add(paramA, in0)
	b0 := Add(a0, paramB)
	out0 := Add(b0, paramsSum)

	// Time step 1
	in1 := NewVariableWithName[T](mat.NewScalar[T](4), true, "Input 1")
	tsh.SetTimeStep(1, in1)

	a1 := Add(paramA, in1)
	b1 := Add(a1, paramB)
	c1 := Add(paramsSum, out0) // note: this is not linked to the input
	out1 := Add(b1, c1)

	// Time step 2
	in2 := NewVariableWithName[T](mat.NewScalar[T](4), true, "Input 2")
	tsh.SetTimeStep(2, in2)

	a2 := Add(paramA, in2)
	b2 := Add(a2, paramB)
	c2 := Add(paramsSum, out0) // note: this is not linked to the input
	out2 := Add(b2, c2)

	fmt.Printf("%#v\n", out2)
	fmt.Printf("%#v\n", tsh)

	assert.Equal(t, -1, tsh.TimeStep(paramA))
	assert.Equal(t, -1, tsh.TimeStep(paramB))
	assert.Equal(t, -1, tsh.TimeStep(paramsSum))

	assert.Equal(t, 0, tsh.TimeStep(in0))
	assert.Equal(t, 0, tsh.TimeStep(a0))
	assert.Equal(t, 0, tsh.TimeStep(b0))
	assert.Equal(t, 0, tsh.TimeStep(out0))

	assert.Equal(t, 1, tsh.TimeStep(in1))
	assert.Equal(t, 1, tsh.TimeStep(a1))
	assert.Equal(t, 1, tsh.TimeStep(b1))
	assert.Equal(t, 1, tsh.TimeStep(c1))
	assert.Equal(t, 1, tsh.TimeStep(out1))

	assert.Equal(t, 2, tsh.TimeStep(in2))
	assert.Equal(t, 2, tsh.TimeStep(a2))
	assert.Equal(t, 2, tsh.TimeStep(b2))
	assert.Equal(t, 2, tsh.TimeStep(c2))
	assert.Equal(t, 2, tsh.TimeStep(out2))
}

func TestTimeStepHandler_SetTimeStep(t *testing.T) {
	t.Run("float32", testTimeStepHandlerSetTimeStep[float32])
	t.Run("float64", testTimeStepHandlerSetTimeStep[float64])
}

func testTimeStepHandlerSetTimeStep[T mat.DType](t *testing.T) {
	t.Run("it panics with a negative time step", func(t *testing.T) {
		tsh := NewTimeStepHandler[T]()
		n := NewVariable[T](mat.NewScalar[T](1), true)
		assert.Panics(t, func() { tsh.SetTimeStep(-1, n) })
	})

	t.Run("it allows duplicate nodes", func(t *testing.T) {
		tsh := NewTimeStepHandler[T]()
		n := NewVariable[T](mat.NewScalar[T](1), true)
		assert.NotPanics(t, func() { tsh.SetTimeStep(0, n, n) })
	})

	t.Run("it panics if called twice with the same time step", func(t *testing.T) {
		tsh := NewTimeStepHandler[T]()
		a := NewVariable[T](mat.NewScalar[T](1), true)
		b := NewVariable[T](mat.NewScalar[T](2), true)
		tsh.SetTimeStep(0, a)
		assert.Panics(t, func() { tsh.SetTimeStep(0, b) })
	})

	t.Run("it panics if a node already has a different time step", func(t *testing.T) {
		tsh := NewTimeStepHandler[T]()
		n := NewVariable[T](mat.NewScalar[T](1), true)
		tsh.SetTimeStep(0, n)
		assert.Panics(t, func() { tsh.SetTimeStep(1, n) })
	})
}
