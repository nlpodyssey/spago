// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

var _ Matrix[float32] = &Dense[float32]{}
var _ Matrix[float64] = &Dense[float64]{}

func TestNewDense(t *testing.T) {
	t.Run("float32", testNewDense[float32])
	t.Run("float64", testNewDense[float64])
}

func testNewDense[T DType](t *testing.T) {
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
				assert.Equal(t, tc.e, d.Data())
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

func testNewVecDense[T DType](t *testing.T) {
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
				assert.Equal(t, tc, d.Data())
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

func testNewScalar[T DType](t *testing.T) {
	d := NewScalar(T(42))
	assertDenseDims(t, 1, 1, d)
	assert.Equal(t, []T{42}, d.Data())
}

func TestNewEmptyVecDense(t *testing.T) {
	t.Run("float32", testNewEmptyVecDense[float32])
	t.Run("float64", testNewEmptyVecDense[float64])
}

func testNewEmptyVecDense[T DType](t *testing.T) {
	t.Run("negative size", func(t *testing.T) {
		require.Panics(t, func() {
			NewEmptyVecDense[T](-1)
		})
	})

	for _, size := range []int{0, 1, 2, 10, 100} {
		t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
			d := NewEmptyVecDense[T](size)
			assertDenseDims(t, size, 1, d)
			for _, v := range d.Data() {
				require.Equal(t, T(0), v)
			}
		})
	}
}

func TestNewEmptyDense(t *testing.T) {
	t.Run("float32", testNewEmptyDense[float32])
	t.Run("float64", testNewEmptyDense[float64])
}

func testNewEmptyDense[T DType](t *testing.T) {
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
				for _, v := range d.Data() {
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

func testNewOneHotVecDense[T DType](t *testing.T) {
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
			assert.Equal(t, tc.d, d.Data())
		})
	}
}

func TestNewInitDense(t *testing.T) {
	t.Run("float32", testNewInitDense[float32])
	t.Run("float64", testNewInitDense[float64])
}

func testNewInitDense[T DType](t *testing.T) {
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
				for _, v := range d.Data() {
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

func testNewInitFuncDense[T DType](t *testing.T) {
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
			assert.Equal(t, tc.d, d.Data())
		})
	}
}

func TestNewInitVecDense(t *testing.T) {
	t.Run("float32", testNewInitVecDense[float32])
	t.Run("float64", testNewInitVecDense[float64])
}

func testNewInitVecDense[T DType](t *testing.T) {
	t.Run("negative size", func(t *testing.T) {
		require.Panics(t, func() {
			NewInitVecDense(-1, T(42))
		})
	})

	for _, size := range []int{0, 1, 2, 10, 100} {
		t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
			d := NewInitVecDense(size, T(42))
			assertDenseDims(t, size, 1, d)
			for _, v := range d.Data() {
				require.Equal(t, T(42), v)
			}
		})
	}
}

func TestNewIdentityDense(t *testing.T) {
	t.Run("float32", testNewIdentityDense[float32])
	t.Run("float64", testNewIdentityDense[float64])
}

func testNewIdentityDense[T DType](t *testing.T) {
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
			assert.Equal(t, tc.d, d.Data())
		})
	}
}

func TestDense_SetData(t *testing.T) {
	t.Run("float32", testDenseSetData[float32])
	t.Run("float64", testDenseSetData[float64])
}

func testDenseSetData[T DType](t *testing.T) {
	t.Run("incompatible data size", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetData([]T{1, 2, 3})
		})
	})

	t.Run("zero size - nil", func(t *testing.T) {
		d := NewEmptyDense[T](0, 0)
		d.SetData(nil)
		assert.Equal(t, []T{}, d.data)
	})

	t.Run("zero size - empty slice", func(t *testing.T) {
		d := NewEmptyDense[T](0, 0)
		d.SetData([]T{})
		assert.Equal(t, []T{}, d.data)
	})

	t.Run("data is set correctly", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		v := []T{1, 2, 3, 7, 8, 9}
		d.SetData(v)
		assert.Equal(t, v, d.data)
	})

	t.Run("data is copied", func(t *testing.T) {
		d := NewEmptyDense[T](1, 1)
		s := []T{1}
		d.SetData(s)
		s[0] = 42 // modifying s must not modify d.data
		assert.Equal(t, T(1), d.data[0])
	})
}

func TestDense_ZerosLike(t *testing.T) {
	t.Run("float32", testDenseZerosLike[float32])
	t.Run("float64", testDenseZerosLike[float64])
}

func testDenseZerosLike[T DType](t *testing.T) {
	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d1 := NewInitDense(r, c, T(42))
				d2 := d1.ZerosLike()
				assertDenseDims(t, r, c, d2.(*Dense[T]))
				for _, v := range d2.Data() {
					require.Equal(t, T(0), v)
				}
			})
		}
	}
}

func TestDense_OnesLike(t *testing.T) {
	t.Run("float32", testDenseOnesLike[float32])
	t.Run("float64", testDenseOnesLike[float64])
}

func testDenseOnesLike[T DType](t *testing.T) {
	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d1 := NewInitDense(r, c, T(42))
				d2 := d1.OnesLike()
				assertDenseDims(t, r, c, d2.(*Dense[T]))
				for _, v := range d2.Data() {
					require.Equal(t, T(1), v)
				}
			})
		}
	}
}

func TestDense_Scalar(t *testing.T) {
	t.Run("float32", testDenseScalar[float32])
	t.Run("float64", testDenseScalar[float64])
}

func testDenseScalar[T DType](t *testing.T) {
	t.Run("non-scalar matrix", func(t *testing.T) {
		d := NewEmptyDense[T](1, 2)
		require.Panics(t, func() {
			d.Scalar()
		})
	})

	t.Run("scalar matrix", func(t *testing.T) {
		d := NewScalar(T(42))
		require.Equal(t, T(42), d.Scalar())
	})
}

func TestDense_Zeros(t *testing.T) {
	t.Run("float32", testDenseZeros[float32])
	t.Run("float64", testDenseZeros[float64])
}

func testDenseZeros[T DType](t *testing.T) {
	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d := NewInitDense(r, c, T(42))
				d.Zeros()
				assertDenseDims(t, r, c, d)
				for _, v := range d.Data() {
					require.Equal(t, T(0), v)
				}
			})
		}
	}
}

func TestDense_Set(t *testing.T) {
	t.Run("float32", testDenseSet[float32])
	t.Run("float64", testDenseSet[float64])
}

func testDenseSet[T DType](t *testing.T) {
	t.Run("negative row", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(-1, 1, 42)
		})
	})

	t.Run("negative col", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(1, -1, 42)
		})
	})

	t.Run("row out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(2, 1, 42)
		})
	})

	t.Run("col out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(1, 3, 42)
		})
	})

	testCases := []struct {
		r    int
		c    int
		setR int
		setC int
		d    []T
	}{
		{1, 1, 0, 0, []T{42}},

		{2, 1, 0, 0, []T{42, 0}},
		{2, 1, 1, 0, []T{0, 42}},

		{1, 2, 0, 0, []T{42, 0}},
		{1, 2, 0, 1, []T{0, 42}},

		{2, 2, 0, 0, []T{
			42, 0,
			0, 0,
		}},
		{2, 2, 0, 1, []T{
			0, 42,
			0, 0,
		}},
		{2, 2, 1, 0, []T{
			0, 0,
			42, 0,
		}},
		{2, 2, 1, 1, []T{
			0, 0,
			0, 42,
		}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d set (%d, %d)", tc.r, tc.c, tc.setR, tc.setC), func(t *testing.T) {
			d := NewEmptyDense[T](tc.r, tc.c)
			d.Set(tc.setR, tc.setC, 42)
			assert.Equal(t, tc.d, d.Data())
		})
	}
}

func TestDense_At(t *testing.T) {
	t.Run("float32", testDenseAt[float32])
	t.Run("float64", testDenseAt[float64])
}

func testDenseAt[T DType](t *testing.T) {
	t.Run("negative row", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.At(-1, 1)
		})
	})

	t.Run("negative col", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.At(1, -1)
		})
	})

	t.Run("row out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.At(2, 1)
		})
	})

	t.Run("col out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.At(1, 3)
		})
	})

	testCases := []struct {
		r   int
		c   int
		atR int
		atC int
		v   T
	}{
		// Each value is a 2-digit number having the format "<row><col>"
		{1, 1, 0, 0, 11},

		{2, 1, 0, 0, 11},
		{2, 1, 1, 0, 21},

		{1, 2, 0, 0, 11},
		{1, 2, 0, 1, 12},

		{2, 2, 0, 0, 11},
		{2, 2, 0, 1, 12},
		{2, 2, 1, 0, 21},
		{2, 2, 1, 1, 22},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d at (%d, %d)", tc.r, tc.c, tc.atR, tc.atC), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.r, tc.c, func(r int, c int) T {
				return T(c + 1 + (r+1)*10)
			})
			v := d.At(tc.atR, tc.atC)
			assert.Equal(t, tc.v, v)
		})
	}
}

func TestDense_SetVec(t *testing.T) {
	t.Run("float32", testDenseSetVec[float32])
	t.Run("float64", testDenseSetVec[float64])
}

func testDenseSetVec[T DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetVec(1, 42)
		})
	})

	t.Run("negative index", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SetVec(-1, 42)
		})
	})

	t.Run("index out of upper bound", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SetVec(2, 42)
		})
	})

	testCases := []struct {
		size int
		i    int
		d    []T
	}{
		{1, 0, []T{42}},
		{2, 0, []T{42, 0}},
		{2, 1, []T{0, 42}},
		{4, 0, []T{42, 0, 0, 0}},
		{4, 1, []T{0, 42, 0, 0}},
		{4, 2, []T{0, 0, 42, 0}},
		{4, 3, []T{0, 0, 0, 42}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("row vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewEmptyDense[T](tc.size, 1)
			d.SetVec(tc.i, 42)
			assert.Equal(t, tc.d, d.Data())
		})

		t.Run(fmt.Sprintf("column vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewEmptyDense[T](1, tc.size)
			d.SetVec(tc.i, 42)
			assert.Equal(t, tc.d, d.Data())
		})
	}
}

func TestDense_AtVec(t *testing.T) {
	t.Run("float32", testDenseAtVec[float32])
	t.Run("float64", testDenseAtVec[float64])
}

func testDenseAtVec[T DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.AtVec(1)
		})
	})

	t.Run("negative index", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.AtVec(-1)
		})
	})

	t.Run("index out of upper bound", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.AtVec(2)
		})
	})

	testCases := []struct {
		size int
		i    int
		v    T
	}{
		{1, 0, 1},
		{2, 0, 1},
		{2, 1, 2},
		{4, 0, 1},
		{4, 1, 2},
		{4, 2, 3},
		{4, 3, 4},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("row vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.size, 1, func(r, _ int) T {
				return T(r + 1)
			})
			v := d.AtVec(tc.i)
			assert.Equal(t, tc.v, v)
		})

		t.Run(fmt.Sprintf("column vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](1, tc.size, func(_, c int) T {
				return T(c + 1)
			})
			v := d.AtVec(tc.i)
			assert.Equal(t, tc.v, v)
		})
	}
}

func TestDense_ExtractRow(t *testing.T) {
	t.Run("float32", testDenseExtractRow[float32])
	t.Run("float64", testDenseExtractRow[float64])
}

func testDenseExtractRow[T DType](t *testing.T) {
	t.Run("negative row", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ExtractRow(-1)
		})
	})

	t.Run("row out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ExtractRow(2)
		})
	})

	testCases := []struct {
		r int
		c int
		i int
		d []T
	}{
		// Each value is a 2-digit number having the format "<row><col>"
		{1, 0, 0, []T{}},
		{1, 1, 0, []T{11}},
		{1, 2, 0, []T{11, 12}},

		{2, 1, 0, []T{11}},
		{2, 1, 1, []T{21}},

		{2, 2, 0, []T{11, 12}},
		{2, 2, 1, []T{21, 22}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d extract %d", tc.r, tc.c, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.r, tc.c, func(r int, c int) T {
				return T(c + 1 + (r+1)*10)
			})
			r := d.ExtractRow(tc.i)
			assertDenseDims(t, len(tc.d), 1, r.(*Dense[T]))
			assert.Equal(t, tc.d, r.Data())
		})
	}
}

func TestDense_ExtractColumn(t *testing.T) {
	t.Run("float32", testDenseExtractColumn[float32])
	t.Run("float64", testDenseExtractColumn[float64])
}

func testDenseExtractColumn[T DType](t *testing.T) {
	t.Run("negative col", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ExtractColumn(-1)
		})
	})

	t.Run("col out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ExtractColumn(3)
		})
	})

	testCases := []struct {
		r int
		c int
		i int
		d []T
	}{
		// Each value is a 2-digit number having the format "<row><col>"
		{0, 1, 0, []T{}},
		{1, 1, 0, []T{11}},
		{2, 1, 0, []T{11, 21}},

		{1, 2, 0, []T{11}},
		{1, 2, 1, []T{12}},

		{2, 2, 0, []T{11, 21}},
		{2, 2, 1, []T{12, 22}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d extract %d", tc.r, tc.c, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.r, tc.c, func(r int, c int) T {
				return T(c + 1 + (r+1)*10)
			})
			c := d.ExtractColumn(tc.i)
			assertDenseDims(t, len(tc.d), 1, c.(*Dense[T]))
			assert.Equal(t, tc.d, c.Data())
		})
	}
}

func TestDense_View(t *testing.T) {
	t.Run("float32", testDenseView[float32])
	t.Run("float64", testDenseView[float64])
}

func testDenseView[T DType](t *testing.T) {
	t.Run("negative rows", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.View(-1, 6)
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.View(6, -1)
		})
	})

	t.Run("incompatible dimensions", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.View(2, 2)
		})
	})

	testCases := []struct {
		r     int
		c     int
		viewR int
		viewC int
	}{
		{0, 0, 0, 0},
		{1, 1, 1, 1},

		{0, 1, 0, 1},
		{0, 1, 1, 0},

		{1, 0, 1, 0},
		{1, 0, 0, 1},

		{1, 2, 1, 2},
		{1, 2, 2, 1},

		{2, 1, 2, 1},
		{2, 1, 1, 2},

		{2, 2, 2, 2},

		// Weird cases, but technically legit
		{2, 2, 1, 4},
		{2, 2, 4, 1},
		{2, 3, 2, 3},
		{2, 3, 3, 2},
		{2, 3, 1, 6},
		{2, 3, 6, 1},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d view %d x %d", tc.r, tc.c, tc.viewR, tc.viewC), func(t *testing.T) {
			d := NewEmptyDense[T](tc.r, tc.c)
			v := d.View(tc.viewR, tc.viewC)
			assertDenseDims(t, tc.viewR, tc.viewC, v.(*Dense[T]))
			assert.Equal(t, d.Data(), v.Data())
		})
	}

	t.Run("data is not copied", func(t *testing.T) {
		d := NewEmptyDense[T](1, 1)
		v := d.View(1, 1)
		d.Set(0, 0, 42) // modifying d must modify v too
		assert.Equal(t, T(42), v.At(0, 0))
		v.Set(0, 0, 2) // modifying v must modify d too
		assert.Equal(t, T(2), d.At(0, 0))
	})
}

func TestDense_Reshape(t *testing.T) {
	t.Run("float32", testDenseReshape[float32])
	t.Run("float64", testDenseReshape[float64])
}

func testDenseReshape[T DType](t *testing.T) {
	t.Run("negative rows", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Reshape(-1, 6)
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Reshape(6, -1)
		})
	})

	t.Run("incompatible dimensions", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Reshape(2, 2)
		})
	})

	testCases := []struct {
		r     int
		c     int
		reshR int
		reshC int
	}{
		{0, 0, 0, 0},
		{1, 1, 1, 1},

		{0, 1, 0, 1},
		{0, 1, 1, 0},

		{1, 0, 1, 0},
		{1, 0, 0, 1},

		{1, 2, 1, 2},
		{1, 2, 2, 1},

		{2, 1, 2, 1},
		{2, 1, 1, 2},

		{2, 2, 2, 2},

		// Weird cases, but technically legit
		{2, 2, 1, 4},
		{2, 2, 4, 1},
		{2, 3, 2, 3},
		{2, 3, 3, 2},
		{2, 3, 1, 6},
		{2, 3, 6, 1},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d reshape %d x %d", tc.r, tc.c, tc.reshR, tc.reshC), func(t *testing.T) {
			d := NewEmptyDense[T](tc.r, tc.c)
			r := d.Reshape(tc.reshR, tc.reshC)
			assertDenseDims(t, tc.reshR, tc.reshC, r.(*Dense[T]))
			assert.Equal(t, d.Data(), r.Data())
		})
	}

	t.Run("data is copied", func(t *testing.T) {
		d := NewEmptyDense[T](1, 1)
		r := d.Reshape(1, 1)
		d.Set(0, 0, 42) // modifying d must not modify r
		assert.Equal(t, T(0), r.At(0, 0))
	})
}

func TestDense_ReshapeInPlace(t *testing.T) {
	t.Run("float32", testDenseReshapeInPlace[float32])
	t.Run("float64", testDenseReshapeInPlace[float64])
}

func testDenseReshapeInPlace[T DType](t *testing.T) {
	t.Run("negative rows", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ReshapeInPlace(-1, 6)
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ReshapeInPlace(6, -1)
		})
	})

	t.Run("incompatible dimensions", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ReshapeInPlace(2, 2)
		})
	})

	testCases := []struct {
		r     int
		c     int
		reshR int
		reshC int
	}{
		{0, 0, 0, 0},
		{1, 1, 1, 1},

		{0, 1, 0, 1},
		{0, 1, 1, 0},

		{1, 0, 1, 0},
		{1, 0, 0, 1},

		{1, 2, 1, 2},
		{1, 2, 2, 1},

		{2, 1, 2, 1},
		{2, 1, 1, 2},

		{2, 2, 2, 2},

		// Weird cases, but technically legit
		{2, 2, 1, 4},
		{2, 2, 4, 1},
		{2, 3, 2, 3},
		{2, 3, 3, 2},
		{2, 3, 1, 6},
		{2, 3, 6, 1},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d reshape %d x %d", tc.r, tc.c, tc.reshR, tc.reshC), func(t *testing.T) {
			d := NewEmptyDense[T](tc.r, tc.c)
			d2 := d.ReshapeInPlace(tc.reshR, tc.reshC)
			assert.Same(t, d, d2)
			assertDenseDims(t, tc.reshR, tc.reshC, d)
		})
	}
}

func TestDense_ResizeVector(t *testing.T) {
	t.Run("float32", testDenseResizeVector[float32])
	t.Run("float64", testDenseResizeVector[float64])
}

func testDenseResizeVector[T DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ResizeVector(2)
		})
	})

	t.Run("negative size", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.ResizeVector(-1)
		})
	})

	testCases := []struct {
		size    int
		newSize int
		d       []T
	}{
		{0, 0, []T{}},

		{1, 0, []T{}},
		{1, 1, []T{1}},
		{1, 2, []T{1, 0}},
		{1, 3, []T{1, 0, 0}},

		{2, 0, []T{}},
		{2, 1, []T{1}},
		{2, 2, []T{1, 2}},
		{2, 3, []T{1, 2, 0}},
		{2, 4, []T{1, 2, 0, 0}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("row vector size %d resize %d", tc.size, tc.newSize), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.size, 1, func(r, _ int) T {
				return T(r + 1)
			})
			r := d.ResizeVector(tc.newSize)
			assert.Equal(t, tc.d, r.Data())
		})

		t.Run(fmt.Sprintf("column vector size %d resize %d", tc.size, tc.newSize), func(t *testing.T) {
			d := NewInitFuncDense[T](1, tc.size, func(_, c int) T {
				return T(c + 1)
			})
			r := d.ResizeVector(tc.newSize)
			assert.Equal(t, tc.d, r.Data())
		})
	}

	t.Run("data is copied - smaller size", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		r := d.ResizeVector(1)
		d.Set(0, 0, 42) // modifying d must not modify r
		assert.Equal(t, T(0), r.At(0, 0))
	})

	t.Run("data is copied - bigger size", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		r := d.ResizeVector(3)
		d.Set(0, 0, 42) // modifying d must not modify r
		assert.Equal(t, T(0), r.At(0, 0))
	})
}

func TestDense_T(t *testing.T) {
	t.Run("float32", testDenseT[float32])
	t.Run("float64", testDenseT[float64])
}

func testDenseT[T DType](t *testing.T) {
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
		{1, 2, []T{11, 12}},
		{2, 1, []T{11, 21}},
		{2, 2, []T{
			11, 21,
			12, 22,
		}},
		{2, 3, []T{
			11, 21,
			12, 22,
			13, 23,
		}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.r, tc.c), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.r, tc.c, func(r int, c int) T {
				return T(c + 1 + (r+1)*10)
			})
			tr := d.T()
			assertDenseDims(t, tc.c, tc.r, tr.(*Dense[T]))
			assert.Equal(t, tc.d, tr.Data())
		})
	}
}

type addTestCase[T DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func addTestCases[T DType]() []addTestCase[T] {
	return []addTestCase[T]{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{2}), NewDense[T](1, 1, []T{10}), []T{12}},
		{
			NewDense[T](1, 2, []T{2, 3}),
			NewDense[T](1, 2, []T{10, 20}),
			[]T{12, 23},
		},
		{
			NewDense[T](1, 2, []T{2, 3}),   // row vec
			NewDense[T](2, 1, []T{10, 20}), // col vec
			[]T{12, 23},
		},
		{
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 7,
			}),
			NewDense[T](2, 3, []T{
				10, 20, 30,
				40, 50, 60,
			}),
			[]T{
				12, 23, 34,
				45, 56, 67,
			},
		},
	}
}

func TestDense_Add(t *testing.T) {
	t.Run("float32", testDenseAdd[float32])
	t.Run("float64", testDenseAdd[float64])
}

func testDenseAdd[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.Add(b)
		})
	})

	for _, tc := range addTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			y := tc.a.Add(tc.b)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_AddInPlace(t *testing.T) {
	t.Run("float32", testDenseAddInPlace[float32])
	t.Run("float64", testDenseAddInPlace[float64])
}

func testDenseAddInPlace[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.AddInPlace(b)
		})
	})

	for _, tc := range addTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			a2 := tc.a.AddInPlace(tc.b)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, tc.a.Data())
		})
	}
}

type addScalarTestCase[T DType] struct {
	a *Dense[T]
	n T
	y []T
}

func addScalarTestCases[T DType]() []addScalarTestCase[T] {
	return []addScalarTestCase[T]{
		{NewEmptyDense[T](0, 0), 10, []T{}},
		{NewEmptyDense[T](0, 1), 10, []T{}},
		{NewEmptyDense[T](1, 0), 10, []T{}},
		{NewDense[T](1, 1, []T{2}), 10, []T{12}},
		{NewDense[T](1, 2, []T{2, 3}), 10, []T{12, 13}},
		{
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 7,
			}),
			10,
			[]T{
				12, 13, 14,
				15, 16, 17,
			},
		},
	}
}

func TestDense_AddScalar(t *testing.T) {
	t.Run("float32", testDenseAddScalar[float32])
	t.Run("float64", testDenseAddScalar[float64])
}

func testDenseAddScalar[T DType](t *testing.T) {
	for _, tc := range addScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			y := tc.a.AddScalar(tc.n)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_AddScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseAddScalarInPlace[float32])
	t.Run("float64", testDenseAddScalarInPlace[float64])
}

func testDenseAddScalarInPlace[T DType](t *testing.T) {
	for _, tc := range addScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			a2 := tc.a.AddScalarInPlace(tc.n)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, tc.a.Data())
		})
	}
}

type subTestCase[T DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func subTestCases[T DType]() []subTestCase[T] {
	return []subTestCase[T]{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{10}), NewDense[T](1, 1, []T{2}), []T{8}},
		{
			NewDense[T](1, 2, []T{10, 20}),
			NewDense[T](1, 2, []T{2, 3}),
			[]T{8, 17},
		},
		{
			NewDense[T](1, 2, []T{10, 20}), // row vec
			NewDense[T](2, 1, []T{2, 3}),   // col vec
			[]T{8, 17},
		},
		{
			NewDense[T](2, 3, []T{
				10, 20, 30,
				40, 50, 60,
			}),
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 7,
			}),
			[]T{
				8, 17, 26,
				35, 44, 53,
			},
		},
	}
}

func TestDense_Sub(t *testing.T) {
	t.Run("float32", testDenseSub[float32])
	t.Run("float64", testDenseSub[float64])
}

func testDenseSub[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.Sub(b)
		})
	})

	for _, tc := range subTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			y := tc.a.Sub(tc.b)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_SubInPlace(t *testing.T) {
	t.Run("float32", testDenseSubInPlace[float32])
	t.Run("float64", testDenseSubInPlace[float64])
}

func testDenseSubInPlace[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.SubInPlace(b)
		})
	})

	for _, tc := range subTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			a2 := tc.a.SubInPlace(tc.b)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, tc.a.Data())
		})
	}
}

type subScalarTestCase[T DType] struct {
	a *Dense[T]
	n T
	y []T
}

func subScalarTestCases[T DType]() []subScalarTestCase[T] {
	return []subScalarTestCase[T]{
		{NewEmptyDense[T](0, 0), 10, []T{}},
		{NewEmptyDense[T](0, 1), 10, []T{}},
		{NewEmptyDense[T](1, 0), 10, []T{}},
		{NewDense[T](1, 1, []T{10}), 2, []T{8}},
		{NewDense[T](1, 2, []T{10, 20}), 2, []T{8, 18}},
		{
			NewDense[T](2, 3, []T{
				10, 20, 30,
				40, 50, 60,
			}),
			2,
			[]T{
				8, 18, 28,
				38, 48, 58,
			},
		},
	}
}

func TestDense_SubScalar(t *testing.T) {
	t.Run("float32", testDenseSubScalar[float32])
	t.Run("float64", testDenseSubScalar[float64])
}

func testDenseSubScalar[T DType](t *testing.T) {
	for _, tc := range subScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			y := tc.a.SubScalar(tc.n)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_SubScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseSubScalarInPlace[float32])
	t.Run("float64", testDenseSubScalarInPlace[float64])
}

func testDenseSubScalarInPlace[T DType](t *testing.T) {
	for _, tc := range subScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			a2 := tc.a.SubScalarInPlace(tc.n)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, tc.a.Data())
		})
	}
}

type prodTestCase[T DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func prodTestCases[T DType]() []prodTestCase[T] {
	return []prodTestCase[T]{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{2}), NewDense[T](1, 1, []T{10}), []T{20}},
		{
			NewDense[T](1, 2, []T{2, 3}),
			NewDense[T](1, 2, []T{10, 20}),
			[]T{20, 60},
		},
		{
			NewDense[T](1, 2, []T{2, 3}),   // row vec
			NewDense[T](2, 1, []T{10, 20}), // col vec
			[]T{20, 60},
		},
		{
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 7,
			}),
			NewDense[T](2, 3, []T{
				10, 20, 30,
				40, 50, 60,
			}),
			[]T{
				20, 60, 120,
				200, 300, 420,
			},
		},
	}
}

func TestDense_Prod(t *testing.T) {
	t.Run("float32", testDenseProd[float32])
	t.Run("float64", testDenseProd[float64])
}

func testDenseProd[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.Prod(b)
		})
	})

	for _, tc := range prodTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			y := tc.a.Prod(tc.b)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_ProdInPlace(t *testing.T) {
	t.Run("float32", testDenseProdInPlace[float32])
	t.Run("float64", testDenseProdInPlace[float64])
}

func testDenseProdInPlace[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.ProdInPlace(b)
		})
	})

	for _, tc := range prodTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			a2 := tc.a.ProdInPlace(tc.b)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, tc.a.Data())
		})
	}
}

type prodScalarTestCase[T DType] struct {
	a *Dense[T]
	n T
	y []T
}

func prodScalarTestCases[T DType]() []prodScalarTestCase[T] {
	return []prodScalarTestCase[T]{
		{NewEmptyDense[T](0, 0), 10, []T{}},
		{NewEmptyDense[T](0, 1), 10, []T{}},
		{NewEmptyDense[T](1, 0), 10, []T{}},
		{NewDense[T](1, 1, []T{2}), 10, []T{20}},
		{
			NewDense[T](1, 2, []T{2, 3}),
			10,
			[]T{20, 30},
		},
		{
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 7,
			}),
			10,
			[]T{
				20, 30, 40,
				50, 60, 70,
			},
		},
	}
}

func TestDense_ProdScalar(t *testing.T) {
	t.Run("float32", testDenseProdScalar[float32])
	t.Run("float64", testDenseProdScalar[float64])
}

func testDenseProdScalar[T DType](t *testing.T) {
	for _, tc := range prodScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			y := tc.a.ProdScalar(tc.n)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_ProdScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseProdScalarInPlace[float32])
	t.Run("float64", testDenseProdScalarInPlace[float64])
}

func testDenseProdScalarInPlace[T DType](t *testing.T) {
	for _, tc := range prodScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			a2 := tc.a.ProdScalarInPlace(tc.n)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, tc.a.Data())
		})
	}
}

func TestDense_ProdMatrixScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseProdMatrixScalarInPlace[float32])
	t.Run("float64", testDenseProdMatrixScalarInPlace[float64])
}

func testDenseProdMatrixScalarInPlace[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.ProdMatrixScalarInPlace(b, 1)
		})
	})

	for _, tc := range prodScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			// start with a "dirty" matrix to ensure it's correctly overwritten
			// and initial data is irrelevant
			y := tc.a.OnesLike()
			y.ProdMatrixScalarInPlace(tc.a, tc.n)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

type divTestCase[T DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func divTestCases[T DType]() []divTestCase[T] {
	return []divTestCase[T]{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{10}), NewDense[T](1, 1, []T{2}), []T{5}},
		{
			NewDense[T](1, 2, []T{10, 20}),
			NewDense[T](1, 2, []T{2, 5}),
			[]T{5, 4},
		},
		{
			NewDense[T](1, 2, []T{10, 20}), // row vec
			NewDense[T](2, 1, []T{2, 5}),   // col vec
			[]T{5, 4},
		},
		{
			NewDense[T](2, 3, []T{
				10, 20, 30,
				40, 50, 60,
			}),
			NewDense[T](2, 3, []T{
				2, 5, 3,
				5, 5, 4,
			}),
			[]T{
				5, 4, 10,
				8, 10, 15,
			},
		},
	}
}

func TestDense_Div(t *testing.T) {
	t.Run("float32", testDenseDiv[float32])
	t.Run("float64", testDenseDiv[float64])
}

func testDenseDiv[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.Div(b)
		})
	})

	for _, tc := range divTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			y := tc.a.Div(tc.b)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_DivInPlace(t *testing.T) {
	t.Run("float32", testDenseDivInPlace[float32])
	t.Run("float64", testDenseDivInPlace[float64])
}

func testDenseDivInPlace[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 4)
		require.Panics(t, func() {
			a.DivInPlace(b)
		})
	})

	for _, tc := range divTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			a2 := tc.a.DivInPlace(tc.b)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, tc.a.Data())
		})
	}
}

func TestDense_Mul(t *testing.T) {
	t.Run("float32", testDenseMul[float32])
	t.Run("float64", testDenseMul[float64])
}

func testDenseMul[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			a.Mul(b)
		})
	})

	testCases := []struct {
		a *Dense[T]
		b *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](0, 1), []T{0}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](1, 0), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](1, 2), []T{}},
		{NewEmptyDense[T](2, 1), NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{2}), NewDense[T](1, 1, []T{10}), []T{20}},
		{NewEmptyDense[T](2, 0), NewEmptyDense[T](0, 3), []T{
			0, 0, 0,
			0, 0, 0,
		}},
		{
			NewDense[T](1, 1, []T{2}),
			NewDense[T](1, 2, []T{10, 20}),
			[]T{20, 40},
		},
		{
			NewDense[T](2, 1, []T{2, 3}),
			NewDense[T](1, 1, []T{10}),
			[]T{20, 30},
		},
		{
			NewDense[T](2, 2, []T{
				2, 3,
				4, 5,
			}),
			NewDense[T](2, 2, []T{
				6, 7,
				8, 9,
			}),
			[]T{
				36, 41,
				64, 73,
			},
		},
		{
			NewDense[T](2, 3, []T{
				2, 3, 4,
				5, 6, 7,
			}),
			NewDense[T](3, 4, []T{
				10, 20, 30, 40,
				50, 60, 70, 80,
				9, 8, 7, 6,
			}),
			[]T{
				206, 252, 298, 344,
				413, 516, 619, 722,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			y := tc.a.Mul(tc.b)
			assertDenseDims(t, tc.a.rows, tc.b.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_MulT(t *testing.T) {
	t.Run("float32", testDenseMulT[float32])
	t.Run("float64", testDenseMulT[float64])
}

func testDenseMulT[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](3, 1)
		require.Panics(t, func() {
			a.MulT(b)
		})
	})

	t.Run("other matrix with zero columns", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 0)
		require.Panics(t, func() {
			a.MulT(b)
		})
	})

	t.Run("other matrix with more than one columns", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			a.MulT(b)
		})
	})

	testCases := []struct {
		a *Dense[T]
		b *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), []T{0}},
		{NewEmptyDense[T](0, 2), NewEmptyDense[T](0, 1), []T{0, 0}},
		{NewEmptyDense[T](0, 2), NewEmptyDense[T](0, 1), []T{0, 0}},
		{NewDense[T](1, 1, []T{2}), NewDense[T](1, 1, []T{10}), []T{20}},
		{
			NewDense[T](1, 2, []T{2, 3}),
			NewDense[T](1, 1, []T{10}),
			[]T{20, 30},
		},
		{
			NewDense[T](2, 2, []T{
				2, 3,
				4, 5,
			}),
			NewDense[T](2, 1, []T{
				6,
				7,
			}),
			[]T{
				40,
				53,
			},
		},
		{
			NewDense[T](3, 2, []T{
				2, 3,
				4, 5,
				6, 7,
			}),
			NewDense[T](3, 1, []T{
				10,
				20,
				30,
			}),
			[]T{
				280,
				340,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			y := tc.a.MulT(tc.b)
			assertDenseDims(t, tc.a.cols, 1, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_DotUnitary(t *testing.T) {
	t.Run("float32", testDenseDotUnitary[float32])
	t.Run("float64", testDenseDotUnitary[float64])
}

func testDenseDotUnitary[T DType](t *testing.T) {
	t.Run("receiver matrix is non-vector", func(t *testing.T) {
		a := NewEmptyDense[T](2, 2)
		b := NewEmptyVecDense[T](4)
		require.Panics(t, func() {
			a.DotUnitary(b)
		})
	})

	t.Run("other matrix is non-vector", func(t *testing.T) {
		a := NewEmptyVecDense[T](4)
		b := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			a.DotUnitary(b)
		})
	})

	t.Run("incompatible data size", func(t *testing.T) {
		a := NewEmptyVecDense[T](2)
		b := NewEmptyVecDense[T](3)
		require.Panics(t, func() {
			a.DotUnitary(b)
		})
	})

	testCases := []struct {
		a *Dense[T]
		b *Dense[T]
		v T
	}{
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), 0},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](1, 0), 0},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), 0},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](0, 1), 0},
		{NewDense[T](1, 1, []T{2}), NewDense[T](1, 1, []T{10}), 20},
		{
			NewDense[T](1, 2, []T{2, 3}),
			NewDense[T](1, 2, []T{10, 20}),
			80,
		},
		{
			NewDense[T](1, 2, []T{2, 3}),   // row vec
			NewDense[T](2, 1, []T{10, 20}), // col vec
			80,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d, %d x %d", tc.a.rows, tc.a.cols, tc.b.rows, tc.b.cols), func(t *testing.T) {
			v := tc.a.DotUnitary(tc.b)
			assert.Equal(t, tc.v, v)
		})
	}
}

func TestDense_ClipInPlace(t *testing.T) {
	t.Run("float32", testDenseClipInPlace[float32])
	t.Run("float64", testDenseClipInPlace[float64])
}

func testDenseClipInPlace[T DType](t *testing.T) {
	t.Run("max < min", func(t *testing.T) {
		d := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			d.ClipInPlace(2, 1)
		})
	})

	testCases := []struct {
		d        *Dense[T]
		min      T
		max      T
		expected []T
	}{
		{NewEmptyDense[T](0, 0), 0, 0, []T{}},
		{NewEmptyDense[T](0, 1), 0, 0, []T{}},
		{NewEmptyDense[T](1, 0), 0, 0, []T{}},
		{NewDense[T](1, 1, []T{2}), 1, 3, []T{2}},
		{NewDense[T](1, 1, []T{2}), 2, 2, []T{2}},
		{NewDense[T](1, 1, []T{2}), 1, 1, []T{1}},
		{NewDense[T](1, 1, []T{2}), 3, 3, []T{3}},
		{
			NewDense[T](2, 4, []T{
				0, 1, 2, 3,
				4, 5, 6, 7,
			}),
			2, 5,
			[]T{
				2, 2, 2, 3,
				4, 5, 5, 5,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d min %g max %g", tc.d.rows, tc.d.cols, tc.min, tc.max), func(t *testing.T) {
			d2 := tc.d.ClipInPlace(tc.min, tc.max)
			assert.Same(t, tc.d, d2)
			assert.Equal(t, tc.expected, tc.d.data)
		})
	}
}

func TestDense_Maximum(t *testing.T) {
	t.Run("float32", testDenseMaximum[float32])
	t.Run("float64", testDenseMaximum[float64])
}

func testDenseMaximum[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			a.Maximum(b)
		})
	})

	testCases := []struct {
		a *Dense[T]
		b *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{2}), NewDense[T](1, 1, []T{3}), []T{3}},
		{
			NewDense[T](1, 3, []T{10, 2, 100}),
			NewDense[T](1, 3, []T{1, 20, 100}),
			[]T{10, 20, 100},
		},
		{
			NewDense[T](2, 3, []T{
				1, 3, 5,
				7, 9, 0,
			}),
			NewDense[T](2, 3, []T{
				0, 4, 4,
				6, 10, 1,
			}),
			[]T{
				1, 4, 5,
				7, 10, 1,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.a.rows, tc.a.cols), func(t *testing.T) {
			y := tc.a.Maximum(tc.b)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_Minimum(t *testing.T) {
	t.Run("float32", testDenseMinimum[float32])
	t.Run("float64", testDenseMinimum[float64])
}

func testDenseMinimum[T DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			a.Minimum(b)
		})
	})

	testCases := []struct {
		a *Dense[T]
		b *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{2}), NewDense[T](1, 1, []T{3}), []T{2}},
		{
			NewDense[T](1, 3, []T{10, 2, 100}),
			NewDense[T](1, 3, []T{1, 20, 100}),
			[]T{1, 2, 100},
		},
		{
			NewDense[T](2, 3, []T{
				1, 3, 5,
				7, 9, 0,
			}),
			NewDense[T](2, 3, []T{
				0, 4, 4,
				6, 10, 1,
			}),
			[]T{
				0, 3, 4,
				6, 9, 0,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.a.rows, tc.a.cols), func(t *testing.T) {
			y := tc.a.Minimum(tc.b)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_Abs(t *testing.T) {
	t.Run("float32", testDenseAbs[float32])
	t.Run("float64", testDenseAbs[float64])
}

func testDenseAbs[T DType](t *testing.T) {
	testCases := []struct {
		d *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{-42}), []T{42}},
		{
			NewDense[T](1, 2, []T{-3, 4}),
			[]T{3, 4},
		},
		{
			NewDense[T](2, 3, []T{
				1, -2, 3,
				-4, 5, -6,
			}),
			[]T{
				1, 2, 3,
				4, 5, 6,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Abs()
			assertDenseDims(t, tc.d.rows, tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_Pow(t *testing.T) {
	t.Run("float32", testDensePow[float32])
	t.Run("float64", testDensePow[float64])
}

func testDensePow[T DType](t *testing.T) {
	testCases := []struct {
		d   *Dense[T]
		pow T
		y   []T
	}{
		{NewEmptyDense[T](0, 0), 2, []T{}},
		{NewEmptyDense[T](0, 1), 2, []T{}},
		{NewEmptyDense[T](1, 0), 2, []T{}},
		{NewDense[T](1, 1, []T{2}), 3, []T{8}},
		{NewDense[T](1, 1, []T{2}), 0, []T{1}},
		{NewDense[T](1, 2, []T{2, -3}), 2, []T{4, 9}},
		{
			NewDense[T](2, 3, []T{
				0, -1, 2,
				-3, 4, -5,
			}),
			3,
			[]T{
				0, -1, 8,
				-27, 64, -125,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d pow %g", tc.d.rows, tc.d.cols, tc.pow), func(t *testing.T) {
			y := tc.d.Pow(tc.pow)
			assertDenseDims(t, tc.d.rows, tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_Sqrt(t *testing.T) {
	t.Run("float32", testDenseSqrt[float32])
	t.Run("float64", testDenseSqrt[float64])
}

func testDenseSqrt[T DType](t *testing.T) {
	testCases := []struct {
		d *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{4}), []T{2}},
		{
			NewDense[T](1, 2, []T{4, 9}),
			[]T{2, 3},
		},
		{
			NewDense[T](2, 3, []T{
				0, 1, 4,
				9, 16, 25,
			}),
			[]T{
				0, 1, 2,
				3, 4, 5,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Sqrt()
			assertDenseDims(t, tc.d.rows, tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_Sum(t *testing.T) {
	t.Run("float32", testDenseSum[float32])
	t.Run("float64", testDenseSum[float64])
}

func testDenseSum[T DType](t *testing.T) {
	testCases := []struct {
		d *Dense[T]
		y T
	}{
		{NewEmptyDense[T](0, 0), 0},
		{NewEmptyDense[T](0, 1), 0},
		{NewEmptyDense[T](1, 0), 0},
		{NewDense[T](1, 1, []T{2}), 2},
		{NewDense[T](1, 2, []T{3, -1}), 2},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			21,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Sum()
			assert.Equal(t, tc.y, y)
		})
	}
}

func TestDense_Max(t *testing.T) {
	t.Run("float32", testDenseMax[float32])
	t.Run("float64", testDenseMax[float64])
}

func testDenseMax[T DType](t *testing.T) {
	t.Run("empty data", func(t *testing.T) {
		d := NewEmptyDense[T](0, 1)
		require.Panics(t, func() {
			d.Max()
		})
	})

	testCases := []struct {
		d *Dense[T]
		y T
	}{
		{NewDense[T](1, 1, []T{2}), 2},
		{NewDense[T](1, 2, []T{3, -1}), 3},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				9, 8, 7,
			}),
			9,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Max()
			assert.Equal(t, tc.y, y)
		})
	}
}

func TestDense_Min(t *testing.T) {
	t.Run("float32", testDenseMin[float32])
	t.Run("float64", testDenseMin[float64])
}

func testDenseMin[T DType](t *testing.T) {
	t.Run("empty data", func(t *testing.T) {
		d := NewEmptyDense[T](0, 1)
		require.Panics(t, func() {
			d.Min()
		})
	})

	testCases := []struct {
		d *Dense[T]
		y T
	}{
		{NewDense[T](1, 1, []T{2}), 2},
		{NewDense[T](1, 2, []T{3, -1}), -1},
		{
			NewDense[T](2, 3, []T{
				3, 2, 1,
				9, 8, 7,
			}),
			1,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Min()
			assert.Equal(t, tc.y, y)
		})
	}
}

func TestDense_Range(t *testing.T) {
	t.Run("float32", testDenseRange[float32])
	t.Run("float64", testDenseRange[float64])
}

func testDenseRange[T DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Range(1, 2)
		})
	})

	t.Run("invalid range", func(t *testing.T) {
		d := NewEmptyVecDense[T](3)
		require.Panics(t, func() {
			d.Range(2, 1)
		})
	})

	t.Run("negative start", func(t *testing.T) {
		d := NewEmptyVecDense[T](3)
		require.Panics(t, func() {
			d.Range(-1, 1)
		})
	})

	t.Run("negative end", func(t *testing.T) {
		d := NewEmptyVecDense[T](3)
		require.Panics(t, func() {
			d.Range(1, -1)
		})
	})

	testCases := []struct {
		size  int
		start int
		end   int
		y     []T
	}{
		{0, 0, 0, []T{}},

		{1, 0, 0, []T{}},
		{1, 0, 1, []T{1}},

		{2, 0, 0, []T{}},
		{2, 0, 1, []T{1}},
		{2, 0, 2, []T{1, 2}},
		{2, 1, 2, []T{2}},
		{2, 1, 1, []T{}},

		{3, 0, 2, []T{1, 2}},
		{3, 1, 3, []T{2, 3}},
		{3, 0, 3, []T{1, 2, 3}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("row vector size %v range %d, %d", tc.size, tc.start, tc.end), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.size, 1, func(r, _ int) T {
				return T(r + 1)
			})
			y := d.Range(tc.start, tc.end)
			assertDenseDims(t, len(tc.y), 1, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})

		t.Run(fmt.Sprintf("column vector size %v range %d, %d", tc.size, tc.start, tc.end), func(t *testing.T) {
			d := NewInitFuncDense[T](1, tc.size, func(_, c int) T {
				return T(c + 1)
			})
			y := d.Range(tc.start, tc.end)
			assertDenseDims(t, len(tc.y), 1, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_SplitV(t *testing.T) {
	t.Run("float32", testDenseSplitV[float32])
	t.Run("float64", testDenseSplitV[float64])
}

func testDenseSplitV[T DType](t *testing.T) {
	t.Run("negative size", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SplitV(-1)
		})
	})

	t.Run("empty sizes", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		y := d.SplitV()
		assert.Nil(t, y)
	})

	testCases := []struct {
		d     *Dense[T]
		sizes []int
		y     [][]T
	}{
		{NewEmptyDense[T](0, 0), []int{0}, [][]T{{}}},
		{NewEmptyDense[T](0, 0), []int{0, 0}, [][]T{{}, {}}},

		{NewEmptyDense[T](0, 1), []int{0}, [][]T{{}}},
		{NewEmptyDense[T](0, 1), []int{0, 0}, [][]T{{}, {}}},

		{NewEmptyDense[T](1, 0), []int{0}, [][]T{{}}},
		{NewEmptyDense[T](1, 0), []int{0, 0}, [][]T{{}, {}}},

		{NewEmptyDense[T](1, 1), []int{0}, [][]T{{}}},
		{NewEmptyDense[T](1, 1), []int{0, 0}, [][]T{{}, {}}},

		{NewDense[T](1, 1, []T{1}), []int{1}, [][]T{{1}}},
		{NewDense[T](1, 1, []T{1}), []int{0, 1}, [][]T{{}, {1}}},
		{NewDense[T](1, 1, []T{1}), []int{1, 0}, [][]T{{1}, {}}},

		{
			NewDense[T](3, 2, []T{
				1, 2,
				3, 4,
				5, 6,
			}),
			[]int{2},
			[][]T{{1, 2}},
		},
		{
			NewDense[T](3, 2, []T{
				1, 2,
				3, 4,
				5, 6,
			}),
			[]int{2, 2, 2},
			[][]T{{1, 2}, {3, 4}, {5, 6}},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d sizes %v", tc.d.rows, tc.d.cols, tc.sizes), func(t *testing.T) {
			y := tc.d.SplitV(tc.sizes...)
			require.Len(t, y, len(tc.y))
			for i, v := range y {
				expectedData := tc.y[i]
				assertDenseDims(t, len(expectedData), 1, v.(*Dense[T]))
				assert.Equal(t, expectedData, v.Data())
			}
		})
	}
}

func TestDense_Augment(t *testing.T) {
	t.Run("float32", testDenseAugment[float32])
	t.Run("float64", testDenseAugment[float64])
}

func testDenseAugment[T DType](t *testing.T) {
	t.Run("non square matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Augment()
		})
	})

	testCases := []struct {
		d *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), []T{}},
		{NewDense[T](1, 1, []T{42}), []T{42, 1}},
		{
			NewDense[T](2, 2, []T{
				1, 2,
				3, 4,
			}),
			[]T{
				1, 2, 1, 0,
				3, 4, 0, 1,
			},
		},
		{
			NewDense[T](3, 3, []T{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}),
			[]T{
				1, 2, 3, 1, 0, 0,
				4, 5, 6, 0, 1, 0,
				7, 8, 9, 0, 0, 1,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Augment()
			assertDenseDims(t, tc.d.rows, tc.d.cols*2, y.(*Dense[T]))
			require.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_SwapInPlace(t *testing.T) {
	t.Run("float32", testDenseSwapInPlace[float32])
	t.Run("float64", testDenseSwapInPlace[float64])
}

func testDenseSwapInPlace[T DType](t *testing.T) {
	t.Run("negative r1", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SwapInPlace(-1, 1)
		})
	})

	t.Run("r1 out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SwapInPlace(2, 1)
		})
	})

	t.Run("negative r2", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SwapInPlace(1, -1)
		})
	})

	t.Run("r2 out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SwapInPlace(1, 2)
		})
	})

	testCases := []struct {
		d  *Dense[T]
		r1 int
		r2 int
		y  []T
	}{
		{NewEmptyDense[T](1, 0), 0, 0, []T{}},
		{NewDense[T](1, 1, []T{1}), 0, 0, []T{1}},
		{NewDense[T](1, 2, []T{1, 2}), 0, 0, []T{1, 2}},
		{
			NewDense[T](2, 1, []T{
				1,
				2,
			}),
			0, 0,
			[]T{
				1,
				2,
			},
		},
		{
			NewDense[T](2, 1, []T{
				1,
				2,
			}),
			0, 1,
			[]T{
				2,
				1,
			},
		},
		{
			NewDense[T](2, 1, []T{
				1,
				2,
			}),
			1, 0,
			[]T{
				2,
				1,
			},
		},
		{
			NewDense[T](3, 2, []T{
				1, 2,
				3, 4,
				5, 6,
			}),
			0, 2,
			[]T{
				5, 6,
				3, 4,
				1, 2,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d swap %d, %d", tc.d.rows, tc.d.cols, tc.r1, tc.r2), func(t *testing.T) {
			d2 := tc.d.SwapInPlace(tc.r1, tc.r2)
			assert.Same(t, tc.d, d2)
			assert.Equal(t, tc.y, tc.d.data)
		})
	}
}

func TestDense_PadRows(t *testing.T) {
	t.Run("float32", testDensePadRows[float32])
	t.Run("float64", testDensePadRows[float64])
}

func testDensePadRows[T DType](t *testing.T) {
	t.Run("negative n", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.PadRows(-1)
		})
	})

	testCases := []struct {
		d *Dense[T]
		n int
		y []T
	}{
		{NewEmptyDense[T](0, 0), 0, []T{}},
		{NewEmptyDense[T](0, 0), 1, []T{}},
		{NewEmptyDense[T](0, 0), 2, []T{}},

		{NewEmptyDense[T](1, 0), 0, []T{}},
		{NewEmptyDense[T](1, 0), 1, []T{}},
		{NewEmptyDense[T](1, 0), 2, []T{}},

		{NewEmptyDense[T](0, 1), 0, []T{}},
		{NewEmptyDense[T](0, 1), 1, []T{0}},
		{NewEmptyDense[T](0, 1), 2, []T{0, 0}},

		{NewDense[T](1, 1, []T{1}), 0, []T{1}},
		{NewDense[T](1, 1, []T{1}), 1, []T{1, 0}},
		{NewDense[T](1, 1, []T{1}), 2, []T{1, 0, 0}},

		{
			NewDense[T](1, 2, []T{
				1, 2,
			}),
			0,
			[]T{
				1, 2,
			},
		},
		{
			NewDense[T](1, 2, []T{
				1, 2,
			}),
			1,
			[]T{
				1, 2,
				0, 0,
			},
		},
		{
			NewDense[T](1, 2, []T{
				1, 2,
			}),
			2,
			[]T{
				1, 2,
				0, 0,
				0, 0,
			},
		},

		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			0,
			[]T{
				1, 2, 3,
				4, 5, 6,
			},
		},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			1,
			[]T{
				1, 2, 3,
				4, 5, 6,
				0, 0, 0,
			},
		},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			2,
			[]T{
				1, 2, 3,
				4, 5, 6,
				0, 0, 0,
				0, 0, 0,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d pad %d", tc.d.rows, tc.d.cols, tc.n), func(t *testing.T) {
			y := tc.d.PadRows(tc.n)
			assertDenseDims(t, tc.d.rows+tc.n, tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_PadColumns(t *testing.T) {
	t.Run("float32", testDensePadColumns[float32])
	t.Run("float64", testDensePadColumns[float64])
}

func testDensePadColumns[T DType](t *testing.T) {
	t.Run("negative n", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.PadColumns(-1)
		})
	})

	testCases := []struct {
		d *Dense[T]
		n int
		y []T
	}{
		{NewEmptyDense[T](0, 0), 0, []T{}},
		{NewEmptyDense[T](0, 0), 1, []T{}},
		{NewEmptyDense[T](0, 0), 2, []T{}},

		{NewEmptyDense[T](0, 1), 0, []T{}},
		{NewEmptyDense[T](0, 1), 1, []T{}},
		{NewEmptyDense[T](0, 1), 2, []T{}},

		{NewEmptyDense[T](1, 0), 0, []T{}},
		{NewEmptyDense[T](1, 0), 1, []T{0}},
		{NewEmptyDense[T](1, 0), 2, []T{0, 0}},

		{NewDense[T](1, 1, []T{1}), 0, []T{1}},
		{NewDense[T](1, 1, []T{1}), 1, []T{1, 0}},
		{NewDense[T](1, 1, []T{1}), 2, []T{1, 0, 0}},

		{
			NewDense[T](2, 1, []T{
				1,
				2,
			}),
			0,
			[]T{
				1,
				2,
			},
		},
		{
			NewDense[T](2, 1, []T{
				1,
				2,
			}),
			1,
			[]T{
				1, 0,
				2, 0,
			},
		},
		{
			NewDense[T](2, 1, []T{
				1,
				2,
			}),
			2,
			[]T{
				1, 0, 0,
				2, 0, 0,
			},
		},

		{
			NewDense[T](3, 2, []T{
				1, 2,
				3, 4,
				5, 6,
			}),
			0,
			[]T{
				1, 2,
				3, 4,
				5, 6,
			},
		},
		{
			NewDense[T](3, 2, []T{
				1, 2,
				3, 4,
				5, 6,
			}),
			1,
			[]T{
				1, 2, 0,
				3, 4, 0,
				5, 6, 0,
			},
		},
		{
			NewDense[T](3, 2, []T{
				1, 2,
				3, 4,
				5, 6,
			}),
			2,
			[]T{
				1, 2, 0, 0,
				3, 4, 0, 0,
				5, 6, 0, 0,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d pad %d", tc.d.rows, tc.d.cols, tc.n), func(t *testing.T) {
			y := tc.d.PadColumns(tc.n)
			assertDenseDims(t, tc.d.rows, tc.d.cols+tc.n, y.(*Dense[T]))
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestDense_Norm(t *testing.T) {
	t.Run("float32", testDenseNorm[float32])
	t.Run("float64", testDenseNorm[float64])
}

func testDenseNorm[T DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Norm(2)
		})
	})

	testCases := []struct {
		x   []T
		pow T
		y   T
	}{
		{[]T{}, 2, 0},
		{[]T{1}, 2, 1},
		{[]T{1, 2}, 2, 2.23607},
		{[]T{1, 2}, 3, 2.08008},
		{[]T{1, 2, 3}, 2, 3.74166},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("row vector %v norm pow %g", tc.x, tc.pow), func(t *testing.T) {
			d := NewDense[T](len(tc.x), 1, tc.x)
			y := d.Norm(tc.pow)
			assert.InDelta(t, tc.y, y, 1.0e-04)
		})

		t.Run(fmt.Sprintf("column vector %v norm pow %g", tc.x, tc.pow), func(t *testing.T) {
			d := NewDense[T](1, len(tc.x), tc.x)
			y := d.Norm(tc.pow)
			assert.InDelta(t, tc.y, y, 1.0e-04)
		})
	}
}

// TODO: TestDense_Pivoting
// TODO: TestDense_Normalize2
// TODO: TestDense_LU
// TODO: TestDense_Inverse
// TODO: TestDense_Apply
// TODO: TestDense_ApplyInPlace
// TODO: TestDense_ApplyWithAlpha
// TODO: TestDense_ApplyWithAlphaInPlace
// TODO: TestDense_DoNonZero
// TODO: TestDense_DoVecNonZero
// TODO: TestDense_Clone
// TODO: TestDense_Copy
// TODO: TestDense_String

func assertDenseDims[T DType](t *testing.T, expectedRows, expectedCols int, d *Dense[T]) {
	t.Helper()

	expectedSize := expectedRows * expectedCols
	dimsRows, dimsCols := d.Dims()

	assert.NotNil(t, d)
	assert.Equal(t, expectedRows, d.Rows())
	assert.Equal(t, expectedRows, dimsRows)
	assert.Equal(t, expectedCols, d.Columns())
	assert.Equal(t, expectedCols, dimsCols)
	assert.Equal(t, expectedSize, d.Size())
	assert.Len(t, d.Data(), expectedSize)
}
