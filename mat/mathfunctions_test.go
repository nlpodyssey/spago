// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestSmallestNonzero(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		assert.Equal(t, float32(math.SmallestNonzeroFloat32), SmallestNonzero[float32]())
	})
	t.Run("float64", func(t *testing.T) {
		assert.Equal(t, math.SmallestNonzeroFloat64, SmallestNonzero[float64]())
	})
}

func TestPi(t *testing.T) {
	t.Run("float32", testPi[float32])
	t.Run("float64", testPi[float64])
}

func testPi[T DType](t *testing.T) {
	assert.Equal(t, T(math.Pi), Pi[T]())
}

func TestPow(t *testing.T) {
	t.Run("float32", testPow[float32])
	t.Run("float64", testPow[float64])
}

func testPow[T DType](t *testing.T) {
	assert.InDelta(t, T(8), Pow[T](2, 3), 1e-10)
}

func TestCos(t *testing.T) {
	t.Run("float32", testCos[float32])
	t.Run("float64", testCos[float64])
}

func testCos[T DType](t *testing.T) {
	assert.InDelta(t, T(-1), Cos[T](Pi[T]()), 1e-10)
}

func TestSin(t *testing.T) {
	t.Run("float32", testSin[float32])
	t.Run("float64", testSin[float64])
}

func testSin[T DType](t *testing.T) {
	assert.InDelta(t, T(1), Sin[T](Pi[T]()/2), 1e-10)
}

func TestCosh(t *testing.T) {
	t.Run("float32", testCosh[float32])
	t.Run("float64", testCosh[float64])
}

func testCosh[T DType](t *testing.T) {
	assert.InDelta(t, 11.59195, Cosh[T](Pi[T]()), 1e-5)
}

func TestSinh(t *testing.T) {
	t.Run("float32", testSinh[float32])
	t.Run("float64", testSinh[float64])
}

func testSinh[T DType](t *testing.T) {
	assert.InDelta(t, 11.54874, Sinh[T](Pi[T]()), 1e-5)
}

func TestExp(t *testing.T) {
	t.Run("float32", testExp[float32])
	t.Run("float64", testExp[float64])
}

func testExp[T DType](t *testing.T) {
	assert.InDelta(t, 2.71828, Exp[T](1), 1e-5)
}

func TestAbs(t *testing.T) {
	t.Run("float32", testAbs[float32])
	t.Run("float64", testAbs[float64])
}

func testAbs[T DType](t *testing.T) {
	assert.Equal(t, T(42), Abs[T](42))
	assert.Equal(t, T(42), Abs[T](-42))
}

func TestSqrt(t *testing.T) {
	t.Run("float32", testSqrt[float32])
	t.Run("float64", testSqrt[float64])
}

func testSqrt[T DType](t *testing.T) {
	assert.InDelta(t, T(3), Sqrt[T](9), 1e-10)
}

func TestLog(t *testing.T) {
	t.Run("float32", testLog[float32])
	t.Run("float64", testLog[float64])
}

func testLog[T DType](t *testing.T) {
	assert.InDelta(t, 0.69314, Log[T](2), 1e-5)
}

func TestTan(t *testing.T) {
	t.Run("float32", testTan[float32])
	t.Run("float64", testTan[float64])
}

func testTan[T DType](t *testing.T) {
	assert.InDelta(t, T(1), Tan[T](Pi[T]()/4), 1e-10)
}

func TestTanh(t *testing.T) {
	t.Run("float32", testTanh[float32])
	t.Run("float64", testTanh[float64])
}

func testTanh[T DType](t *testing.T) {
	assert.InDelta(t, 0.76159, Tanh[T](1), 1e-5)
}

func TestMax(t *testing.T) {
	t.Run("float32", testMax[float32])
	t.Run("float64", testMax[float64])
}

func testMax[T DType](t *testing.T) {
	assert.Equal(t, T(2), Max[T](1, 2))
	assert.Equal(t, T(2), Max[T](2, 1))
}

func TestInf(t *testing.T) {
	t.Run("float32", testInf[float32])
	t.Run("float64", testInf[float64])
}

func testInf[T DType](t *testing.T) {
	assert.True(t, math.IsInf(float64(Inf[T](1)), +1))
	assert.True(t, math.IsInf(float64(Inf[T](-1)), -1))
}

func TestIsInf(t *testing.T) {
	t.Run("float32", testIsInf[float32])
	t.Run("float64", testIsInf[float64])
}

func testIsInf[T DType](t *testing.T) {
	assert.True(t, IsInf(Inf[T](1), +1))
	assert.True(t, IsInf(Inf[T](-1), -1))
	assert.False(t, IsInf(Inf[T](-1), +1))
	assert.False(t, IsInf(Inf[T](1), -1))
	assert.False(t, IsInf(T(0), 1))
	assert.False(t, IsInf(T(0), -1))
}

func TestNaN(t *testing.T) {
	t.Run("float32", testNaN[float32])
	t.Run("float64", testNaN[float64])
}

func testNaN[T DType](t *testing.T) {
	assert.True(t, math.IsNaN(float64(NaN[T]())))
}

func TestCeil(t *testing.T) {
	t.Run("float32", testCeil[float32])
	t.Run("float64", testCeil[float64])
}

func testCeil[T DType](t *testing.T) {
	assert.Equal(t, T(2), Ceil[T](1.2))
}

func TestFloor(t *testing.T) {
	t.Run("float32", testFloor[float32])
	t.Run("float64", testFloor[float64])
}

func testFloor[T DType](t *testing.T) {
	assert.Equal(t, T(1), Floor[T](1.9))
}

func TestRound(t *testing.T) {
	t.Run("float32", testRound[float32])
	t.Run("float64", testRound[float64])
}

func testRound[T DType](t *testing.T) {
	assert.Equal(t, T(1), Round[T](1.4))
	assert.Equal(t, T(2), Round[T](1.5))
	assert.Equal(t, T(2), Round[T](1.6))
}
