// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTimeStepHandler(t *testing.T) {
	t.Run("float32", testTimeStepHandler[float32])
	t.Run("float64", testTimeStepHandler[float64])
}

func testTimeStepHandler[T float.DType](t *testing.T) {
	tsh := NewTimeStepHandler()
	require.Equal(t, 0, tsh.CurrentTimeStep())

	// Simulate some parameters, with no associated time step (default 0)
	paramA := Var(mat.NewScalar[T](1)).WithGrad(true).WithName("Param 0")
	paramB := Var(mat.NewScalar[T](2)).WithGrad(true).WithName("Param 1")

	// Perform an operation while still on initial time step 0
	paramsSum := Sum(paramA, paramB)

	// Time step 1
	tsh.IncTimeStep()
	require.Equal(t, 1, tsh.CurrentTimeStep())
	in1 := Var(mat.NewScalar[T](3)).WithGrad(true).WithName("Input 0")

	a1 := Add(paramA, in1)
	b1 := Add(a1, paramB)
	out1 := Add(b1, paramsSum)

	// Time step 2
	tsh.IncTimeStep()
	require.Equal(t, 2, tsh.CurrentTimeStep())
	in2 := Var(mat.NewScalar[T](4)).WithGrad(true).WithName("Input 1")

	a2 := Add(paramA, in2)
	b2 := Add(a2, paramB)
	c2 := Add(paramsSum, out1) // note: this is not linked to the input
	out2 := Add(b2, c2)

	// Time step 2
	tsh.IncTimeStep()
	in3 := Var(mat.NewScalar[T](4)).WithGrad(true).WithName("Input 3")

	a3 := Add(paramA, in3)
	b3 := Add(a3, paramB)
	c3 := Add(paramsSum, out2) // note: this is not linked to the input
	out3 := Add(b3, c3)

	assert.Equal(t, 0, NodeTimeStep(tsh, paramA))
	assert.Equal(t, 0, NodeTimeStep(tsh, paramB))
	assert.Equal(t, 0, NodeTimeStep(tsh, paramsSum))

	assert.Equal(t, 1, NodeTimeStep(tsh, in1))
	assert.Equal(t, 1, NodeTimeStep(tsh, a1))
	assert.Equal(t, 1, NodeTimeStep(tsh, b1))
	assert.Equal(t, 1, NodeTimeStep(tsh, out1))

	assert.Equal(t, 2, NodeTimeStep(tsh, in2))
	assert.Equal(t, 2, NodeTimeStep(tsh, a2))
	assert.Equal(t, 2, NodeTimeStep(tsh, b2))
	assert.Equal(t, 2, NodeTimeStep(tsh, c2))
	assert.Equal(t, 2, NodeTimeStep(tsh, out2))

	assert.Equal(t, 3, NodeTimeStep(tsh, in3))
	assert.Equal(t, 3, NodeTimeStep(tsh, a3))
	assert.Equal(t, 3, NodeTimeStep(tsh, b3))
	assert.Equal(t, 3, NodeTimeStep(tsh, c3))
	assert.Equal(t, 3, NodeTimeStep(tsh, out3))
}
