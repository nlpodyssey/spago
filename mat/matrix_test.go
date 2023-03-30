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
			d := NewEmptyDense[T](tc.r, tc.c)
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
			d := NewEmptyDense[T](tc.r, tc.c)
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
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](3, 2)
		assert.False(t, SameDims(a, b))
		assert.False(t, SameDims(b, a))
	})

	t.Run("same dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 3)
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
		var d Matrix = NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			ConcatV[T](d)
		})
	})

	testCases := []struct {
		xs []Matrix
		y  []T
	}{
		{[]Matrix{}, []T{}},
		{[]Matrix{NewEmptyDense[T](0, 1)}, []T{}},
		{[]Matrix{NewEmptyDense[T](1, 0)}, []T{}},
		{[]Matrix{NewDense[T](1, 1, []T{1})}, []T{1}},
		{
			[]Matrix{
				NewDense[T](1, 1, []T{1}),
				NewDense[T](1, 1, []T{2}),
			},
			[]T{1, 2},
		},
		{
			[]Matrix{
				NewDense[T](1, 2, []T{1, 2}),
				NewDense[T](2, 1, []T{3, 4}),
			},
			[]T{1, 2, 3, 4},
		},
		{
			[]Matrix{
				NewDense[T](1, 1, []T{1}),
				NewDense[T](2, 1, []T{2, 3}),
				NewDense[T](1, 3, []T{4, 5, 6}),
			},
			[]T{1, 2, 3, 4, 5, 6},
		},
	}

	for _, tc := range testCases {
		name := "["
		for _, x := range tc.xs {
			name += fmt.Sprintf(" (%d x %d)", x.Rows(), x.Columns())
		}
		name += " ]"
		t.Run(name, func(t *testing.T) {
			y := ConcatV[T](tc.xs...)
			assert.Equal(t, len(tc.y), y.Rows())
			assert.Equal(t, 1, y.Columns())
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
		var d Matrix = NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			Stack[T](d)
		})
	})

	t.Run("vectors of different sizes", func(t *testing.T) {
		var a Matrix = NewEmptyDense[T](1, 2)
		var b Matrix = NewEmptyDense[T](1, 3)
		require.Panics(t, func() {
			Stack[T](a, b)
		})
	})

	testCases := []struct {
		xs []Matrix
		y  []T
	}{
		{[]Matrix{}, []T{}},
		{[]Matrix{NewEmptyDense[T](0, 1)}, []T{}},
		{[]Matrix{NewEmptyDense[T](1, 0)}, []T{}},
		{[]Matrix{NewDense[T](1, 1, []T{1})}, []T{1}},
		{
			[]Matrix{
				NewDense[T](1, 1, []T{1}),
				NewDense[T](1, 1, []T{2}),
			},
			[]T{
				1,
				2,
			},
		},
		{
			[]Matrix{
				NewDense[T](2, 1, []T{1, 2}),
				NewDense[T](2, 1, []T{3, 4}),
			},
			[]T{
				1, 2,
				3, 4,
			},
		},
		{
			[]Matrix{
				NewDense[T](2, 1, []T{1, 2}),
				NewDense[T](1, 2, []T{3, 4}),
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
			name += fmt.Sprintf(" (%d x %d)", x.Rows(), x.Columns())
		}
		name += " ]"
		t.Run(name, func(t *testing.T) {
			y := Stack[T](tc.xs...)
			assert.Equal(t, len(tc.xs), y.Rows())
			cols := 0
			if len(tc.xs) > 0 {
				cols = tc.xs[0].Size()
			}
			assert.Equal(t, cols, y.Columns())
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
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), true},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), true},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), true},
		{NewEmptyDense[T](1, 1), NewEmptyDense[T](1, 2), false},
		{NewEmptyDense[T](1, 1), NewEmptyDense[T](2, 1), false},
		{NewEmptyDense[T](1, 2), NewEmptyDense[T](2, 1), false},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{42}), true},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{41}), false},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			true,
		},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 7,
			}),
			false,
		},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("Equal(%dx%d, %dx%d) == %v",
			tc.a.Rows(), tc.a.Columns(), tc.b.Rows(), tc.b.Columns(), tc.expected)
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
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), 0, true},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), 0, true},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), 0, true},
		{NewEmptyDense[T](1, 1), NewEmptyDense[T](1, 2), 0, false},
		{NewEmptyDense[T](1, 1), NewEmptyDense[T](2, 1), 0, false},
		{NewEmptyDense[T](1, 2), NewEmptyDense[T](2, 1), 0, false},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{42}), 0, true},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{42.1}), 0, false},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{42.09}), .1, true},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{43}), 1, true},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{44}), 2, true},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{44.1}), 2, false},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 7,
			}),
			1,
			true,
		},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 8,
			}),
			1,
			false,
		},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("InDelta(%dx%d, %dx%d, delta %.1f) == %v",
			tc.a.Rows(), tc.a.Columns(), tc.b.Rows(), tc.b.Columns(), tc.delta, tc.expected)
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, InDelta(tc.a, tc.b, tc.delta), "a vs b")
			assert.Equal(t, tc.expected, InDelta(tc.b, tc.a, tc.delta), "b vs a")
		})
	}
}

func TestCopyValue(t *testing.T) {
	t.Run("float32", testCopyValue[float32])
	t.Run("float64", testCopyValue[float64])
}

func testCopyValue[T float.DType](t *testing.T) {
	t.Run("nil value", func(t *testing.T) {
		n := &dummyItem{value: nil}
		v := CopyValue(n)
		assert.Nil(t, v)
	})

	t.Run("matrix value", func(t *testing.T) {
		n := &dummyItem{value: NewScalar[T](42)}
		v := CopyValue(n)
		RequireMatrixEquals(t, n.value, v)
		assert.NotSame(t, n.value, v)
	})
}

func TestCopyValues(t *testing.T) {
	t.Run("float32", testCopyValues[float32])
	t.Run("float64", testCopyValues[float64])
}

type valuer interface {
	Value() Matrix
}
type dummyItem struct {
	value Matrix
	grad  Matrix
}

func (i *dummyItem) Value() Matrix { return i.value }
func (i *dummyItem) Grad() Matrix  { return i.grad }

func testCopyValues[T float.DType](t *testing.T) {
	nodes := []valuer{
		&dummyItem{value: NewScalar[T](1)},
		&dummyItem{},
		NewScalar[T](3),
	}

	vs := CopyValues(nodes)
	require.Len(t, vs, 3)

	RequireMatrixEquals(t, nodes[0].Value(), vs[0])
	assert.NotSame(t, nodes[0].Value(), vs[0])

	assert.Nil(t, vs[1])

	RequireMatrixEquals(t, nodes[2].Value(), vs[2])
	assert.NotSame(t, nodes[2].Value(), vs[2])
}

func TestCopyGrad(t *testing.T) {
	t.Run("float32", testCopyGrad[float32])
	t.Run("float64", testCopyGrad[float64])
}

func testCopyGrad[T float.DType](t *testing.T) {
	t.Run("nil grad", func(t *testing.T) {
		n := &dummyItem{
			grad: nil,
		}
		v := CopyGrad(n)
		assert.Nil(t, v)
	})

	t.Run("matrix grad", func(t *testing.T) {
		n := &dummyItem{
			grad: NewScalar[T](42),
		}
		v := CopyGrad(n)
		RequireMatrixEquals(t, n.grad, v)
		assert.NotSame(t, n.grad, v)
	})
}
