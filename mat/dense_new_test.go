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

func TestNewDense(t *testing.T) {
	t.Run("float32", testNewDense[float32])
	t.Run("float64", testNewDense[float64])
}

func testNewDense[T float.DType](t *testing.T) {
	t.Run("negative rows", func(t *testing.T) {
		require.Panics(t, func() {
			NewDense(-1, 1, []T{})
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		require.Panics(t, func() {
			NewDense(1, -1, []T{})
		})
	})

	t.Run("elements length mismatch", func(t *testing.T) {
		require.Panics(t, func() {
			NewDense(1, 1, []T{1, 2})
		})
	})

	testCases := []struct {
		r int
		c int
		e []T
	}{
		{0, 0, nil},
		{0, 0, []T{}},

		{0, 1, nil},
		{0, 1, []T{}},

		{1, 0, nil},
		{1, 0, []T{}},

		{1, 1, []T{1}},
		{1, 2, []T{1, 2}},
		{2, 1, []T{1, 2}},
		{2, 2, []T{1, 2, 3, 4}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d, %d, %#v", tc.r, tc.c, tc.e), func(t *testing.T) {
			d := NewDense(tc.r, tc.c, tc.e)
			assertDenseDims(t, tc.r, tc.c, d)
			assert.Len(t, d.Data(), len(tc.e))
			if tc.e != nil {
				assert.Equal(t, tc.e, Data[T](d))
			}
		})
	}

	t.Run("data is copied", func(t *testing.T) {
		s := []T{1}
		d := NewDense(1, 1, s)
		s[0] = 42 // modifying s must not modify d.data
		assert.Equal(t, T(1), d.data[0])
	})
}

func TestNewVecDense(t *testing.T) {
	t.Run("float32", testNewVecDense[float32])
	t.Run("float64", testNewVecDense[float64])
}

func testNewVecDense[T float.DType](t *testing.T) {
	testCases := [][]T{
		nil,
		{},
		{1},
		{1, 2},
		{1, 2, 3},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%#v", tc), func(t *testing.T) {
			d := NewVecDense(tc)
			assertDenseDims(t, len(tc), 1, d)
			assert.Len(t, d.Data(), len(tc))
			if tc != nil {
				assert.Equal(t, tc, Data[T](d))
			}
		})
	}

	t.Run("data is copied", func(t *testing.T) {
		s := []T{1}
		d := NewVecDense(s)
		s[0] = 42 // modifying s must not modify d.data
		assert.Equal(t, T(1), d.data[0])
	})
}

func TestNewScalar(t *testing.T) {
	t.Run("float32", testNewScalar[float32])
	t.Run("float64", testNewScalar[float64])
}

func testNewScalar[T float.DType](t *testing.T) {
	d := Scalar(T(42))
	assertDenseDims(t, 1, 1, d)
	assert.Equal(t, []T{42}, Data[T](d))
}

func TestNewEmptyVecDense(t *testing.T) {
	t.Run("float32", testNewEmptyVecDense[float32])
	t.Run("float64", testNewEmptyVecDense[float64])
}

func testNewEmptyVecDense[T float.DType](t *testing.T) {
	t.Run("negative size", func(t *testing.T) {
		require.Panics(t, func() {
			NewEmptyVecDense[T](-1)
		})
	})

	for _, size := range []int{0, 1, 2, 10, 100} {
		t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
			d := NewEmptyVecDense[T](size)
			assertDenseDims(t, size, 1, d)
			for _, v := range Data[T](d) {
				require.Equal(t, T(0), v)
			}
		})
	}
}

func TestNewEmptyDense(t *testing.T) {
	t.Run("float32", testNewEmptyDense[float32])
	t.Run("float64", testNewEmptyDense[float64])
}

func testNewEmptyDense[T float.DType](t *testing.T) {
	t.Run("negative rows", func(t *testing.T) {
		require.Panics(t, func() {
			NewEmptyDense[T](-1, 1)
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		require.Panics(t, func() {
			NewEmptyDense[T](1, -1)
		})
	})

	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d := NewEmptyDense[T](r, c)
				assertDenseDims(t, r, c, d)
				for _, v := range Data[T](d) {
					require.Equal(t, T(0), v)
				}
			})
		}
	}
}

func TestNewOneHotVecDense(t *testing.T) {
	t.Run("float32", testNewOneHotVecDense[float32])
	t.Run("float64", testNewOneHotVecDense[float64])
}

func testNewOneHotVecDense[T float.DType](t *testing.T) {
	t.Run("negative size", func(t *testing.T) {
		require.Panics(t, func() {
			NewOneHotVecDense[T](-1, 0)
		})
	})

	t.Run("zero size", func(t *testing.T) {
		require.Panics(t, func() {
			NewOneHotVecDense[T](0, 0)
		})
	})

	t.Run("oneAt >= size", func(t *testing.T) {
		require.Panics(t, func() {
			NewOneHotVecDense[T](1, 1)
		})
	})

	t.Run("oneAt negative", func(t *testing.T) {
		require.Panics(t, func() {
			NewOneHotVecDense[T](1, -1)
		})
	})

	testCases := []struct {
		s int
		i int
		d []T
	}{
		{1, 0, []T{1}},
		{2, 0, []T{1, 0}},
		{2, 1, []T{0, 1}},
		{3, 0, []T{1, 0, 0}},
		{3, 1, []T{0, 1, 0}},
		{3, 2, []T{0, 0, 1}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d, %d", tc.s, tc.i), func(t *testing.T) {
			d := NewOneHotVecDense[T](tc.s, tc.i)
			assertDenseDims(t, tc.s, 1, d)
			assert.Equal(t, tc.d, Data[T](d))
		})
	}
}

func TestNewInitDense(t *testing.T) {
	t.Run("float32", testNewInitDense[float32])
	t.Run("float64", testNewInitDense[float64])
}

func testNewInitDense[T float.DType](t *testing.T) {
	t.Run("negative rows", func(t *testing.T) {
		require.Panics(t, func() {
			NewInitDense(-1, 1, T(42))
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		require.Panics(t, func() {
			NewInitDense(1, -1, T(42))
		})
	})

	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d := NewInitDense(r, c, T(42))
				assertDenseDims(t, r, c, d)
				for _, v := range Data[T](d) {
					require.Equal(t, T(42), v)
				}
			})
		}
	}
}

func TestNewInitFuncDense(t *testing.T) {
	t.Run("float32", testNewInitFuncDense[float32])
	t.Run("float64", testNewInitFuncDense[float64])
}

func testNewInitFuncDense[T float.DType](t *testing.T) {
	t.Run("negative rows", func(t *testing.T) {
		require.Panics(t, func() {
			NewInitFuncDense[T](-1, 1, func(r int, c int) T {
				t.Fatal("the callback should not be called")
				return 0
			})
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		require.Panics(t, func() {
			NewInitFuncDense[T](1, -1, func(r int, c int) T {
				t.Fatal("the callback should not be called")
				return 0
			})
		})
	})

	testCases := []struct {
		r int
		c int
		d []T
	}{
		// Each value is a 2-digit number having the format "<row><col>"
		{0, 0, []T{}},
		{0, 1, []T{}},
		{1, 0, []T{}},
		{1, 1, []T{11}},
		{2, 1, []T{11, 21}},
		{3, 1, []T{11, 21, 31}},
		{1, 3, []T{11, 12, 13}},
		{2, 2, []T{
			11, 12,
			21, 22,
		}},
		{3, 3, []T{
			11, 12, 13,
			21, 22, 23,
			31, 32, 33,
		}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.r, tc.c), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.r, tc.c, func(r int, c int) T {
				if len(tc.d) == 0 {
					t.Fatal("the callback should not be called")
				}
				return T(c + 1 + (r+1)*10)
			})
			assertDenseDims(t, tc.r, tc.c, d)
			assert.Equal(t, tc.d, Data[T](d))
		})
	}
}

func TestNewInitVecDense(t *testing.T) {
	t.Run("float32", testNewInitVecDense[float32])
	t.Run("float64", testNewInitVecDense[float64])
}

func testNewInitVecDense[T float.DType](t *testing.T) {
	t.Run("negative size", func(t *testing.T) {
		require.Panics(t, func() {
			NewInitVecDense(-1, T(42))
		})
	})

	for _, size := range []int{0, 1, 2, 10, 100} {
		t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
			d := NewInitVecDense(size, T(42))
			assertDenseDims(t, size, 1, d)
			for _, v := range Data[T](d) {
				require.Equal(t, T(42), v)
			}
		})
	}
}

func TestNewIdentityDense(t *testing.T) {
	t.Run("float32", testNewIdentityDense[float32])
	t.Run("float64", testNewIdentityDense[float64])
}

func testNewIdentityDense[T float.DType](t *testing.T) {
	t.Run("negative size", func(t *testing.T) {
		require.Panics(t, func() {
			NewIdentityDense[T](-1)
		})
	})

	testCases := []struct {
		s int
		d []T
	}{
		{0, []T{}},
		{1, []T{1}},
		{2, []T{
			1, 0,
			0, 1,
		}},
		{3, []T{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		}},
		{4, []T{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("size %d", tc.s), func(t *testing.T) {
			d := NewIdentityDense[T](tc.s)
			assertDenseDims(t, tc.s, tc.s, d)
			assert.Equal(t, tc.d, Data[T](d))
		})
	}
}
