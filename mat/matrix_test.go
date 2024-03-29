// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsVector(t *testing.T) {
	t.Run("float32", testIsVector[float32])
	t.Run("float64", testIsVector[float64])
}

func testIsVector[T float.DType](t *testing.T) {
	testCases := []struct {
		r int
		c int
		b bool
	}{
		{0, 0, false},
		{0, 1, true},
		{1, 0, true},
		{1, 1, true},
		{1, 2, true},
		{2, 1, true},
		{1, 9, true},
		{9, 1, true},
		{2, 2, false},
		{3, 4, false},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.r, tc.c), func(t *testing.T) {
			d := NewDense[T](WithShape(tc.r, tc.c))
			require.Equal(t, tc.b, IsVector(d))
		})
	}
}

func TestIsScalar(t *testing.T) {
	t.Run("float32", testIsScalar[float32])
	t.Run("float64", testIsScalar[float64])
}

func testIsScalar[T float.DType](t *testing.T) {
	testCases := []struct {
		r int
		c int
		b bool
	}{
		{0, 0, false},
		{0, 1, false},
		{1, 0, false},
		{1, 1, true},
		{1, 2, false},
		{2, 1, false},
		{2, 2, false},
		{3, 4, false},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.r, tc.c), func(t *testing.T) {
			d := NewDense[T](WithShape(tc.r, tc.c))
			require.Equal(t, tc.b, IsScalar(d))
		})
	}
}

func TestSameDims(t *testing.T) {
	t.Run("float32", testSameDims[float32])
	t.Run("float64", testSameDims[float64])
}

func testSameDims[T float.DType](t *testing.T) {
	t.Run("different dimensions", func(t *testing.T) {
		a := NewDense[T](WithShape(2, 3))
		b := NewDense[T](WithShape(3, 2))
		assert.False(t, SameDims(a, b))
		assert.False(t, SameDims(b, a))
	})

	t.Run("same dimensions", func(t *testing.T) {
		a := NewDense[T](WithShape(2, 3))
		b := NewDense[T](WithShape(2, 3))
		assert.True(t, SameDims(a, b))
		assert.True(t, SameDims(b, a))
	})
}

func TestConcatV(t *testing.T) {
	t.Run("float32", testConcatV[float32])
	t.Run("float64", testConcatV[float64])
}

func testConcatV[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		var d Matrix = NewDense[T](WithShape(2, 3))
		require.Panics(t, func() {
			ConcatV[T](d)
		})
	})

	testCases := []struct {
		xs []Matrix
		y  []T
	}{
		{[]Matrix{}, []T{}},
		{[]Matrix{NewDense[T](WithShape(0, 1))}, []T{}},
		{[]Matrix{NewDense[T](WithShape(1, 0))}, []T{}},
		{[]Matrix{NewDense[T](WithShape(1, 1), WithBacking([]T{1}))}, []T{1}},
		{
			[]Matrix{
				NewDense[T](WithShape(1, 1), WithBacking([]T{1})),
				NewDense[T](WithShape(1, 1), WithBacking([]T{2})),
			},
			[]T{1, 2},
		},
		{
			[]Matrix{
				NewDense[T](WithShape(1, 2), WithBacking([]T{1, 2})),
				NewDense[T](WithShape(2, 1), WithBacking([]T{3, 4})),
			},
			[]T{1, 2, 3, 4},
		},
		{
			[]Matrix{
				NewDense[T](WithShape(1, 1), WithBacking([]T{1})),
				NewDense[T](WithShape(2, 1), WithBacking([]T{2, 3})),
				NewDense[T](WithShape(1, 3), WithBacking([]T{4, 5, 6})),
			},
			[]T{1, 2, 3, 4, 5, 6},
		},
	}

	for _, tc := range testCases {
		name := "["
		for _, x := range tc.xs {
			name += fmt.Sprintf(" (%d x %d)", x.Shape()[0], x.Shape()[1])
		}
		name += " ]"
		t.Run(name, func(t *testing.T) {
			y := ConcatV[T](tc.xs...)
			assert.Equal(t, len(tc.y), y.Shape()[0])
			assert.Equal(t, 1, y.Shape()[1])
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestStack(t *testing.T) {
	t.Run("float32", testStack[float32])
	t.Run("float64", testStack[float64])
}

func testStack[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		var d Matrix = NewDense[T](WithShape(2, 3))
		require.Panics(t, func() {
			Stack[T](d)
		})
	})

	t.Run("vectors of different sizes", func(t *testing.T) {
		var a Matrix = NewDense[T](WithShape(1, 2))
		var b Matrix = NewDense[T](WithShape(1, 3))
		require.Panics(t, func() {
			Stack[T](a, b)
		})
	})

	testCases := []struct {
		xs []Matrix
		y  []T
	}{
		{[]Matrix{}, []T{}},
		{[]Matrix{NewDense[T](WithShape(0, 1))}, []T{}},
		{[]Matrix{NewDense[T](WithShape(1, 0))}, []T{}},
		{[]Matrix{NewDense[T](WithShape(1, 1), WithBacking([]T{1}))}, []T{1}},
		{
			[]Matrix{
				NewDense[T](WithShape(1, 1), WithBacking([]T{1})),
				NewDense[T](WithShape(1, 1), WithBacking([]T{2})),
			},
			[]T{
				1,
				2,
			},
		},
		{
			[]Matrix{
				NewDense[T](WithShape(2, 1), WithBacking([]T{1, 2})),
				NewDense[T](WithShape(2, 1), WithBacking([]T{3, 4})),
			},
			[]T{
				1, 2,
				3, 4,
			},
		},
		{
			[]Matrix{
				NewDense[T](WithShape(2, 1), WithBacking([]T{1, 2})),
				NewDense[T](WithShape(1, 2), WithBacking([]T{3, 4})),
			},
			[]T{
				1, 2,
				3, 4,
			},
		},
	}

	for _, tc := range testCases {
		name := "["
		for _, x := range tc.xs {
			name += fmt.Sprintf(" (%d x %d)", x.Shape()[0], x.Shape()[1])
		}
		name += " ]"
		t.Run(name, func(t *testing.T) {
			y := Stack[T](tc.xs...)
			assert.Equal(t, len(tc.xs), y.Shape()[0])
			cols := 0
			if len(tc.xs) > 0 {
				cols = tc.xs[0].Size()
			}
			assert.Equal(t, cols, y.Shape()[1])
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestEqual(t *testing.T) {
	t.Run("float32", testEqual[float32])
	t.Run("float64", testEqual[float64])
}

func testEqual[T float.DType](t *testing.T) {
	testCases := []struct {
		a, b     Matrix
		expected bool
	}{
		{NewDense[T](WithShape(0, 0)), NewDense[T](WithShape(0, 0)), true},
		{NewDense[T](WithShape(0, 1)), NewDense[T](WithShape(0, 1)), true},
		{NewDense[T](WithShape(1, 0)), NewDense[T](WithShape(1, 0)), true},
		{NewDense[T](WithShape(1, 1)), NewDense[T](WithShape(1, 2)), false},
		{NewDense[T](WithShape(1, 1)), NewDense[T](WithShape(2, 1)), false},
		{NewDense[T](WithShape(1, 2)), NewDense[T](WithShape(2, 1)), false},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{42})), true},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{41})), false},
		{
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				1, 2, 3,
				4, 5, 6,
			})),
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				1, 2, 3,
				4, 5, 6,
			})),
			true,
		},
		{
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				1, 2, 3,
				4, 5, 6,
			})),
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				1, 2, 3,
				4, 5, 7,
			})),
			false,
		},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("Equal(%dx%d, %dx%d) == %v",
			tc.a.Shape()[0], tc.a.Shape()[1], tc.b.Shape()[0], tc.b.Shape()[1], tc.expected)
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, Equal(tc.a, tc.b), "a vs b")
			assert.Equal(t, tc.expected, Equal(tc.b, tc.a), "b vs a")
		})
	}
}

func TestInDelta(t *testing.T) {
	t.Run("float32", testInDelta[float32])
	t.Run("float64", testInDelta[float64])
}

func testInDelta[T float.DType](t *testing.T) {
	testCases := []struct {
		a, b     Matrix
		delta    float64
		expected bool
	}{
		{NewDense[T](WithShape(0, 0)), NewDense[T](WithShape(0, 0)), 0, true},
		{NewDense[T](WithShape(0, 1)), NewDense[T](WithShape(0, 1)), 0, true},
		{NewDense[T](WithShape(1, 0)), NewDense[T](WithShape(1, 0)), 0, true},
		{NewDense[T](WithShape(1, 1)), NewDense[T](WithShape(1, 2)), 0, false},
		{NewDense[T](WithShape(1, 1)), NewDense[T](WithShape(2, 1)), 0, false},
		{NewDense[T](WithShape(1, 2)), NewDense[T](WithShape(2, 1)), 0, false},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{42})), 0, true},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{42.1})), 0, false},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{42.09})), .1, true},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{43})), 1, true},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{44})), 2, true},
		{NewDense[T](WithShape(1, 1), WithBacking([]T{42})), NewDense[T](WithShape(1, 1), WithBacking([]T{44.1})), 2, false},
		{
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				1, 2, 3,
				4, 5, 6,
			})),
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				2, 3, 4,
				5, 6, 7,
			})),
			1,
			true,
		},
		{
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				1, 2, 3,
				4, 5, 6,
			})),
			NewDense[T](WithShape(2, 3), WithBacking([]T{
				2, 3, 4,
				5, 6, 8,
			})),
			1,
			false,
		},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("InDelta(%dx%d, %dx%d, delta %.1f) == %v",
			tc.a.Shape()[0], tc.a.Shape()[1], tc.b.Shape()[0], tc.b.Shape()[1], tc.delta, tc.expected)
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, InDelta(tc.a, tc.b, tc.delta), "a vs b")
			assert.Equal(t, tc.expected, InDelta(tc.b, tc.a, tc.delta), "b vs a")
		})
	}
}
