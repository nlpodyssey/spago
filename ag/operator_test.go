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

var _ mat.Tensor = &Operator{}

func TestNewOperator(t *testing.T) {
	t.Run("float32", testNewOperator[float32])
	t.Run("float64", testNewOperator[float64])
}

func testNewOperator[T float.DType](t *testing.T) {
	forwardResult := mat.Scalar[T](42)

	f := &dummyFunction[T, mat.Tensor]{
		forward: func() (mat.Tensor, error) { return forwardResult, nil },
	}
	op := NewOperator(f).Run()

	require.NotNil(t, op)

	v := op.Value() // also waits for async forwarding
	assert.Same(t, forwardResult, v)
	assert.Equal(t, 1, f.forwardCalls)
}

func TestOperator_Operands(t *testing.T) {
	t.Run("with generics - float32", testOperatorOperands[float32])
	t.Run("with generics - float64", testOperatorOperands[float64])
}

func testOperatorOperands[T float.DType](t *testing.T) {
	operands := []mat.Tensor{mat.NewDense[T](mat.WithBacking([]T{1, 2, 3}))}
	f := &dummyFunction[T, mat.Tensor]{
		operands: func() []mat.Tensor { return operands },
	}
	op := NewOperator(f).Run()
	require.Equal(t, operands, op.Operands())
	assert.Same(t, operands[0], op.Operands()[0])
}

func TestOperator_Value(t *testing.T) {
	t.Run("with generics - float32", testOperatorValue[float32])
	t.Run("with generics - float64", testOperatorValue[float64])
}

func testOperatorValue[T float.DType](t *testing.T) {
	forwardResult := mat.Scalar[T](42)

	f := &dummyFunction[T, mat.Tensor]{
		forward: func() (mat.Tensor, error) { return forwardResult, nil },
	}
	op := NewOperator(f).Run()

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
		op := NewOperator(&dummyFunction[T, mat.Tensor]{}).Run()
		assert.False(t, op.RequiresGrad())
	})

	t.Run("false if no operands require grad", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, mat.Tensor]{
			operands: func() []mat.Tensor {
				return []mat.Tensor{
					mat.NewDense[T](mat.WithBacking([]T{1.0})),
					mat.NewDense[T](mat.WithBacking([]T{2.0})),
				}
			},
		}).Run()
		assert.False(t, op.RequiresGrad())
	})

	t.Run("true if at least one operand requires grad", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, mat.Tensor]{
			operands: func() []mat.Tensor {
				return []mat.Tensor{
					mat.NewDense[T](mat.WithBacking([]T{1.0})),
					mat.NewDense[T](mat.WithBacking([]T{2.0}), mat.WithGrad(true)),
				}
			},
		}).Run()
		assert.True(t, op.RequiresGrad())
	})
}

func TestOperator_Gradients(t *testing.T) {
	t.Run("float32", testOperatorGradients[float32])
	t.Run("float64", testOperatorGradients[float64])
}

func testOperatorGradients[T float.DType](t *testing.T) {
	t.Run("with requires gradient true", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, mat.Tensor]{
			forward: func() (mat.Tensor, error) {
				return mat.Scalar[T](42), nil
			},
			operands: func() []mat.Tensor {
				return []mat.Tensor{mat.NewDense[T](mat.WithBacking([]T{2.0}), mat.WithGrad(true))}
			},
		}).Run()

		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())

		op.AccGrad(mat.Scalar[T](5))
		mat.RequireMatrixEquals(t, mat.Scalar[T](5), op.Grad().(mat.Matrix))
		assert.True(t, op.HasGrad())

		op.AccGrad(mat.Scalar[T](10))
		mat.RequireMatrixEquals(t, mat.Scalar[T](15), op.Grad().(mat.Matrix))
		assert.True(t, op.HasGrad())

		op.ZeroGrad()
		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())
	})

	t.Run("with requires gradient false", func(t *testing.T) {
		op := NewOperator(&dummyFunction[T, mat.Tensor]{
			forward: func() (mat.Tensor, error) { return mat.Scalar[T](42), nil },
		}).Run()

		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())

		op.AccGrad(mat.Scalar[T](5))
		require.NotNil(t, op.Grad())
		assert.True(t, op.HasGrad())

		op.ZeroGrad()
		require.Nil(t, op.Grad())
		assert.False(t, op.HasGrad())
	})
}

type dummyFunction[T float.DType, O mat.Tensor] struct {
	forward       func() (mat.Tensor, error)
	backward      func(gy mat.Tensor) error
	operands      func() []O
	forwardCalls  int
	backwardCalls int
}

func (f *dummyFunction[T, O]) Forward() (mat.Tensor, error) {
	f.forwardCalls++
	if f.forward == nil {
		return mat.NewDense[T](mat.WithShape(0, 0)), nil // since nil values are not allowed
	}
	return f.forward()
}

func (f *dummyFunction[T, O]) Backward(gy mat.Tensor) error {
	f.backwardCalls++
	if f.backward == nil {
		return nil
	}
	return f.backward(gy)
}

func (f *dummyFunction[T, O]) Operands() []O {
	if f.operands == nil {
		return nil
	}
	return f.operands()
}
