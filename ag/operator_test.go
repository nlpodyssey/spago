// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/stretchr/testify/require"
	"testing"

	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestNewOperator(t *testing.T) {
	t.Run("float32", testNewOperator[float32])
	t.Run("float64", testNewOperator[float64])
}

func testNewOperator[T mat.DType](t *testing.T) {
	forwardResult := mat.NewScalar[T](42)

	f := &dummyFunction[T, Node[T]]{
		forward: func() mat.Matrix[T] { return forwardResult },
	}
	op := NewOperator[T](f)

	require.NotNil(t, op)

	v := op.Value() // also waits for async forwarding
	assert.Same(t, forwardResult, v)
	assert.Equal(t, 1, f.forwardCalls)
}

func TestOperator_Name(t *testing.T) {
	t.Run("without generics", func(t *testing.T) {
		op := NewOperator[float32](&dummyFunctionFloat32{})
		assert.Equal(t, "dummyFunctionFloat32", op.Name())
	})

	t.Run("with generics - float32", testOperatorName[float32])
	t.Run("with generics - float64", testOperatorName[float64])
}

func testOperatorName[T mat.DType](t *testing.T) {
	op := NewOperator[T](&dummyFunction[T, Node[T]]{})
	assert.Equal(t, "dummyFunction", op.Name())
}

func TestOperator_Operands(t *testing.T) {
	t.Run("with generics - float32", testOperatorOperands[float32])
	t.Run("with generics - float64", testOperatorOperands[float64])
}

func testOperatorOperands[T mat.DType](t *testing.T) {
	operands := []Node[T]{&dummyNode[T]{id: 1}}
	f := &dummyFunction[T, Node[T]]{
		operands: func() []Node[T] { return operands },
	}
	op := NewOperator[T](f).(*Operator[T])
	require.Equal(t, operands, op.Operands())
	assert.Same(t, operands[0], op.Operands()[0])
}

type dummyFunction[T mat.DType, O fn.Operand[T]] struct {
	forward       func() mat.Matrix[T]
	backward      func(gy mat.Matrix[T])
	operands      func() []O
	forwardCalls  int
	backwardCalls int
}

func (f *dummyFunction[T, O]) Forward() mat.Matrix[T] {
	f.forwardCalls++
	if f.forward == nil {
		return mat.NewEmptyDense[T](0, 0) // since nil values are not allowed
	}
	return f.forward()
}

func (f *dummyFunction[T, O]) Backward(gy mat.Matrix[T]) {
	f.backwardCalls++
	if f.backward == nil {
		return
	}
	f.backward(gy)
}

func (f *dummyFunction[T, O]) Operands() []O {
	if f.operands == nil {
		return nil
	}
	return f.operands()
}

type dummyFunctionFloat32 struct {
	dummyFunction[float32, Node[float32]]
}
