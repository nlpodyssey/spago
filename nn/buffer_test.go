// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/gob"
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/fn"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	_ fn.DualValue = &Buffer{}
	_ ag.Node      = &Buffer{}
)

func TestBuf(t *testing.T) {
	t.Run("float32", testBuf[float32])
	t.Run("float64", testBuf[float64])
}

func testBuf[T float.DType](t *testing.T) {
	v := mat.NewVecDense([]T{1, 2, 3})
	c := Buf(v)
	require.NotNil(t, c)
	assert.Empty(t, c.Name())
	assert.Same(t, v, c.Value())
	assert.Nil(t, c.Grad())
	assert.False(t, c.HasGrad())
	assert.False(t, c.RequiresGrad())
}

func TestConst_WithName(t *testing.T) {
	t.Run("float32", testConstWithName[float32])
	t.Run("float64", testConstWithName[float64])
}

func testConstWithName[T float.DType](t *testing.T) {
	v := mat.NewVecDense([]T{1, 2, 3})
	c := Buf(v).WithName("foo")
	require.NotNil(t, c)
	assert.Equal(t, "foo", c.Name())
	assert.Same(t, v, c.Value())
	assert.Nil(t, c.Grad())
	assert.False(t, c.HasGrad())
	assert.False(t, c.RequiresGrad())
}

func TestNewConstant(t *testing.T) {
	t.Run("float32", testScalarConst[float32])
	t.Run("float64", testScalarConst[float64])
}

func testScalarConst[T float.DType](t *testing.T) {
	c := Const(T(42))
	require.NotNil(t, c)
	assert.Equal(t, "42", c.Name())
	mat.AssertMatrixEquals(t, mat.NewScalar(T(42)), c.Value())
	assert.Nil(t, c.Grad())
	assert.False(t, c.HasGrad())
	assert.False(t, c.RequiresGrad())
}

func TestScalarConst_WithName(t *testing.T) {
	t.Run("float32", testScalarConstWithName[float32])
	t.Run("float64", testScalarConstWithName[float64])
}

func testScalarConstWithName[T float.DType](t *testing.T) {
	c := Const(T(42)).WithName("foo")
	require.NotNil(t, c)
	assert.Equal(t, "foo", c.Name())
	mat.AssertMatrixEquals(t, mat.NewScalar(T(42)), c.Value())
	assert.Nil(t, c.Grad())
	assert.False(t, c.HasGrad())
	assert.False(t, c.RequiresGrad())
}

func TestSConstant_AccGrad(t *testing.T) {
	t.Run("float32", testScalarConstantAccGrad[float32])
	t.Run("float64", testScalarConstantAccGrad[float64])
}

func testScalarConstantAccGrad[T float.DType](t *testing.T) {
	c := Const(T(42))
	c.AccGrad(mat.NewScalar(T(100)))
	mat.AssertMatrixEquals(t, mat.NewScalar(T(42)), c.Value())
	assert.Nil(t, c.Grad())
	assert.False(t, c.HasGrad())
	assert.False(t, c.RequiresGrad())
}

func TestSConstant_ZeroGrad(t *testing.T) {
	t.Run("float32", testScalarConstantZeroGrad[float32])
	t.Run("float64", testScalarConstantZeroGrad[float64])
}

func testScalarConstantZeroGrad[T float.DType](t *testing.T) {
	c := Const(T(42))
	c.ZeroGrad()
	mat.AssertMatrixEquals(t, mat.NewScalar(T(42)), c.Value())
	assert.Nil(t, c.Grad())
	assert.False(t, c.HasGrad())
	assert.False(t, c.RequiresGrad())
}

func TestSConstant_Marshaling(t *testing.T) {
	t.Run("float32", testScalarConstantMarshaling[float32])
	t.Run("float64", testScalarConstantMarshaling[float64])
}

func testScalarConstantMarshaling[T float.DType](t *testing.T) {
	c1 := Const(T(42)).WithName("foo")

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(c1)
	require.Nil(t, err)

	var c2 *Buffer

	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&c2)
	require.Nil(t, err)

	assert.Equal(t, "foo", c2.Name())
	mat.AssertMatrixEquals(t, mat.NewScalar(T(42)), c2.Value())
}
