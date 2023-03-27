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

func TestNewOperator(t *testing.T) {
	t.Run("float32", testNewOperator[float32])
	t.Run("float64", testNewOperator[float64])
}

func testNewOperator[T float.DType](t *testing.T) {
	forwardResult := mat.NewScalar[T](42)

	f := &dummyFunction[T, Node]{
		forward: func() (mat.Matrix, error) { return forwardResult, nil },
	}
	op := NewOperator(f)

	require.NotNil(t, op)

	v := op.Value() // also waits for async forwarding
	assert.Same(t, forwardResult, v)
	assert.Equal(t, 1, f.forwardCalls)
}

/*
func TestOperator_Name(t *testing.T) {
	t.Run("without generics", func(t *testing.T) {
		op := NewOperator(&dummyFunctionFloat32{})
		assert.Equal(t, "dummyFunctionFloat32", op.Name())
	})

	t.Run("with generics - float32", testOperatorName[float32])
	t.Run("with generics - float64", testOperatorName[float64])
}

func testOperatorName[T float.DType](t *testing.T) {
	op := NewOperator(&dummyFunction[T, Node]{})
	assert.Equal(t, "dummyFunction", op.Name())
}
*/

func TestOperator_Operands(t *testing.T) {
	t.Run("with generics - float32", testOperatorOperands[float32])
	t.Run("with generics - float64", testOperatorOperands[float64])
}

func testOperatorOperands[T float.DType](t *testing.T) {
	operands := []Node{&dummyNode{id: 1}}
	f := &dummyFunction[T, Node]{
		operands: func() []Node { return operands },
	}
	op := NewOperator(f).(*Operator)
	require.Equal(t, operands, op.Operands())
	assert.Same(t, operands[0], op.Operands()[0])
}

func TestOperator_Value(t *testing.T) {
	t.Run("with generics - float32", testOperatorValue[float32])
	t.Run("with generics - float64", testOperatorValue[float64])
}

func testOperatorValue[T float.DType](t *testing.T) {
	forwardResult := mat.NewScalar[T](42)

	f := &dummyFunction[T, Node]{
		forward: func() (mat.Matrix, error) { return forwardResult, nil },
	}
	op := NewOperator(f)

	// The first call to ValueOf() waits for the forward and returns the result
	assert.Same(t, forwardResult, op.Value())

	// The second call returns the same cached result
	assert.Same(t, forwardResult, op.Value())

	// The forward function must have been called only once
	assert.Equal(t, 1, f.forwardCalls)
}

func TestOperator_RequiresGrad(t *testing.T) {
	t.Run("with generics - float32", testOperatorRequiresGrad[float32])
	t.Run("with generics - float64", testOperatorRequiresGrad[float64])
}

func testOperatorRequiresGrad[T float.DType](t *testing.T) {
	t.Run("false without operands", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, Node]{})
		assert.False(t, op.RequiresGrad())
	})

	t.Run("false if no operands require grad", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, Node]{
			operands: func() []Node {
				return []Node{
					&dummyNode{id: 1, requiresGrad: false},
					&dummyNode{id: 2, requiresGrad: false},
				}
			},
		})
		assert.False(t, op.RequiresGrad())
	})

	t.Run("true if at least one operand requires grad", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, Node]{
			operands: func() []Node {
				return []Node{
					&dummyNode{id: 1, requiresGrad: false},
					&dummyNode{id: 2, requiresGrad: true},
				}
			},
		})
		assert.True(t, op.RequiresGrad())
	})
}

func TestOperator_Gradients(t *testing.T) {
	t.Run("float32", testOperatorGradients[float32])
	t.Run("float64", testOperatorGradients[float64])
}

func testOperatorGradients[T float.DType](t *testing.T) {
	t.Run("with requires gradient true", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, Node]{
			forward: func() (mat.Matrix, error) {
				return mat.NewScalar[T](42), nil
			},
			operands: func() []Node {
				return []Node{&dummyNode{requiresGrad: true}}
			},
		})

		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())

		op.AccGrad(mat.NewScalar[T](5))
		mat.RequireMatrixEquals(t, mat.NewScalar[T](5), op.Grad())
		assert.True(t, op.HasGrad())

		op.AccGrad(mat.NewScalar[T](10))
		mat.RequireMatrixEquals(t, mat.NewScalar[T](15), op.Grad())
		assert.True(t, op.HasGrad())

		op.ZeroGrad()
		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())
	})

	t.Run("with requires gradient false", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, Node]{
			forward: func() (mat.Matrix, error) { return mat.NewScalar[T](42), nil },
		})

		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())

		op.AccGrad(mat.NewScalar[T](5))
		require.NotNil(t, op.Grad())
		assert.True(t, op.HasGrad())

		op.ZeroGrad()
		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())
	})
}

type dummyFunction[T float.DType, O Node] struct {
	forward       func() (mat.Matrix, error)
	backward      func(gy mat.Matrix) error
	operands      func() []O
	forwardCalls  int
	backwardCalls int
}

func (f *dummyFunction[T, O]) Forward() (mat.Matrix, error) {
	f.forwardCalls++
	if f.forward == nil {
		return mat.NewEmptyDense[T](0, 0), nil // since nil values are not allowed
	}
	return f.forward()
}

func (f *dummyFunction[T, O]) Backward(gy mat.Matrix) error {
	f.backwardCalls++
	if f.backward == nil {
		return nil
	}
	f.backward(gy)
	return nil
}

func (f *dummyFunction[T, O]) Operands() []O {
	if f.operands == nil {
		return nil
	}
	return f.operands()
}

type dummyFunctionFloat32 struct {
	dummyFunction[float32, Node]
}
