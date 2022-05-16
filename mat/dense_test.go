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

var _ Matrix = &Dense[float32]{}
var _ Matrix = &Dense[float64]{}

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
	d := NewScalar(T(42))
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

func TestDense_SetData(t *testing.T) {
	t.Run("float32", testDenseSetData[float32])
	t.Run("float64", testDenseSetData[float64])
}

func testDenseSetData[T float.DType](t *testing.T) {
	t.Run("incompatible data size", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			SetData[T](d, []T{1, 2, 3})
		})
	})

	t.Run("zero size - nil", func(t *testing.T) {
		d := NewEmptyDense[T](0, 0)
		SetData[T](d, nil)
		assert.Equal(t, []T{}, d.data)
	})

	t.Run("zero size - empty slice", func(t *testing.T) {
		d := NewEmptyDense[T](0, 0)
		SetData[T](d, []T{})
		assert.Equal(t, []T{}, d.data)
	})

	t.Run("data is set correctly", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		v := []T{1, 2, 3, 7, 8, 9}
		SetData[T](d, v)
		assert.Equal(t, v, d.data)
	})

	t.Run("data is copied", func(t *testing.T) {
		d := NewEmptyDense[T](1, 1)
		s := []T{1}
		SetData[T](d, s)
		s[0] = 42 // modifying s must not modify d.data
		assert.Equal(t, T(1), d.data[0])
	})
}

func TestDense_ZerosLike(t *testing.T) {
	t.Run("float32", testDenseZerosLike[float32])
	t.Run("float64", testDenseZerosLike[float64])
}

func testDenseZerosLike[T float.DType](t *testing.T) {
	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d1 := NewInitDense(r, c, T(42))
				d2 := d1.ZerosLike()
				assertDenseDims(t, r, c, d2.(*Dense[T]))
				for _, v := range Data[T](d2) {
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

func testDenseOnesLike[T float.DType](t *testing.T) {
	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d1 := NewInitDense(r, c, T(42))
				d2 := d1.OnesLike()
				assertDenseDims(t, r, c, d2.(*Dense[T]))
				for _, v := range Data[T](d2) {
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

func testDenseScalar[T float.DType](t *testing.T) {
	t.Run("non-scalar matrix", func(t *testing.T) {
		d := NewEmptyDense[T](1, 2)
		require.Panics(t, func() {
			d.Scalar()
		})
	})

	t.Run("scalar matrix", func(t *testing.T) {
		d := NewScalar(T(42))
		require.Equal(t, float.Interface(T(42)), d.Scalar())
	})
}

func TestDense_Zeros(t *testing.T) {
	t.Run("float32", testDenseZeros[float32])
	t.Run("float64", testDenseZeros[float64])
}

func testDenseZeros[T float.DType](t *testing.T) {
	for _, r := range []int{0, 1, 2, 10, 100} {
		for _, c := range []int{0, 1, 2, 10, 100} {
			t.Run(fmt.Sprintf("%d x %d", r, c), func(t *testing.T) {
				d := NewInitDense(r, c, T(42))
				d.Zeros()
				assertDenseDims(t, r, c, d)
				for _, v := range Data[T](d) {
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

func testDenseSet[T float.DType](t *testing.T) {
	t.Run("given matrix not 1×1", func(t *testing.T) {
		d := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			d.Set(1, 1, NewEmptyDense[T](1, 2))
		})
	})

	t.Run("negative row", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(-1, 1, NewScalar(T(42)))
		})
	})

	t.Run("negative col", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(1, -1, NewScalar(T(42)))
		})
	})

	t.Run("row out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(2, 1, NewScalar(T(42)))
		})
	})

	t.Run("col out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Set(1, 3, NewScalar(T(42)))
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
			d.Set(tc.setR, tc.setC, NewScalar(T(42)))
			assert.Equal(t, tc.d, Data[T](d))
		})
	}
}

func TestDense_At(t *testing.T) {
	t.Run("float32", testDenseAt[float32])
	t.Run("float64", testDenseAt[float64])
}

func testDenseAt[T float.DType](t *testing.T) {
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
			assert.Equal(t, float.Interface(tc.v), v.Scalar())
		})
	}
}

func TestDense_SetScalar(t *testing.T) {
	t.Run("float32", testDenseSetScalar[float32])
	t.Run("float64", testDenseSetScalar[float64])
}

func testDenseSetScalar[T float.DType](t *testing.T) {
	t.Run("negative row", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetScalar(-1, 1, float.Interface(T(42)))
		})
	})

	t.Run("negative col", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetScalar(1, -1, float.Interface(T(42)))
		})
	})

	t.Run("row out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetScalar(2, 1, float.Interface(T(42)))
		})
	})

	t.Run("col out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetScalar(1, 3, float.Interface(T(42)))
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
			d.SetScalar(tc.setR, tc.setC, float.Interface(T(42)))
			assert.Equal(t, tc.d, Data[T](d))
		})
	}
}

func TestDense_ScalarAt(t *testing.T) {
	t.Run("float32", testDenseScalarAt[float32])
	t.Run("float64", testDenseScalarAt[float64])
}

func testDenseScalarAt[T float.DType](t *testing.T) {
	t.Run("negative row", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ScalarAt(-1, 1)
		})
	})

	t.Run("negative col", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ScalarAt(1, -1)
		})
	})

	t.Run("row out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ScalarAt(2, 1)
		})
	})

	t.Run("col out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ScalarAt(1, 3)
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
			v := d.ScalarAt(tc.atR, tc.atC)
			assert.Equal(t, float.Interface(tc.v), v)
		})
	}
}

func TestDense_SetVec(t *testing.T) {
	t.Run("float32", testDenseSetVec[float32])
	t.Run("float64", testDenseSetVec[float64])
}

func testDenseSetVec[T float.DType](t *testing.T) {
	t.Run("given matrix not 1×1", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SetVec(1, NewEmptyDense[T](1, 2))
		})
	})

	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetVec(1, NewScalar(T(42)))
		})
	})

	t.Run("negative index", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SetVec(-1, NewScalar(T(42)))
		})
	})

	t.Run("index out of upper bound", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SetVec(2, NewScalar(T(42)))
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
		t.Run(fmt.Sprintf("column vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewEmptyDense[T](tc.size, 1)
			d.SetVec(tc.i, NewScalar(T(42)))
			assert.Equal(t, tc.d, Data[T](d))
		})

		t.Run(fmt.Sprintf("row vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewEmptyDense[T](1, tc.size)
			d.SetVec(tc.i, NewScalar(T(42)))
			assert.Equal(t, tc.d, Data[T](d))
		})
	}
}

func TestDense_AtVec(t *testing.T) {
	t.Run("float32", testDenseAtVec[float32])
	t.Run("float64", testDenseAtVec[float64])
}

func testDenseAtVec[T float.DType](t *testing.T) {
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
		t.Run(fmt.Sprintf("column vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.size, 1, func(r, _ int) T {
				return T(r + 1)
			})
			v := d.AtVec(tc.i)
			assert.Equal(t, float.Interface(tc.v), v.Scalar())
		})

		t.Run(fmt.Sprintf("row vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](1, tc.size, func(_, c int) T {
				return T(c + 1)
			})
			v := d.AtVec(tc.i)
			assert.Equal(t, float.Interface(tc.v), v.Scalar())
		})
	}
}

func TestDense_SetVecScalar(t *testing.T) {
	t.Run("float32", testDenseSetVecScalar[float32])
	t.Run("float64", testDenseSetVecScalar[float64])
}

func testDenseSetVecScalar[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SetVecScalar(1, float.Interface(T(42)))
		})
	})

	t.Run("negative index", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SetVecScalar(-1, float.Interface(T(42)))
		})
	})

	t.Run("index out of upper bound", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SetVecScalar(2, float.Interface(T(42)))
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
		t.Run(fmt.Sprintf("column vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewEmptyDense[T](tc.size, 1)
			d.SetVecScalar(tc.i, float.Interface(T(42)))
			assert.Equal(t, tc.d, Data[T](d))
		})

		t.Run(fmt.Sprintf("row vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewEmptyDense[T](1, tc.size)
			d.SetVecScalar(tc.i, float.Interface(T(42)))
			assert.Equal(t, tc.d, Data[T](d))
		})
	}
}

func TestDense_ScalarAtVec(t *testing.T) {
	t.Run("float32", testDenseScalarAtVec[float32])
	t.Run("float64", testDenseScalarAtVec[float64])
}

func testDenseScalarAtVec[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ScalarAtVec(1)
		})
	})

	t.Run("negative index", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.ScalarAtVec(-1)
		})
	})

	t.Run("index out of upper bound", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.ScalarAtVec(2)
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
		t.Run(fmt.Sprintf("column vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.size, 1, func(r, _ int) T {
				return T(r + 1)
			})
			v := d.ScalarAtVec(tc.i)
			assert.Equal(t, float.Interface(tc.v), v)
		})

		t.Run(fmt.Sprintf("row vector size %d set %d", tc.size, tc.i), func(t *testing.T) {
			d := NewInitFuncDense[T](1, tc.size, func(_, c int) T {
				return T(c + 1)
			})
			v := d.ScalarAtVec(tc.i)
			assert.Equal(t, float.Interface(tc.v), v)
		})
	}
}

func TestDense_ExtractRow(t *testing.T) {
	t.Run("float32", testDenseExtractRow[float32])
	t.Run("float64", testDenseExtractRow[float64])
}

func testDenseExtractRow[T float.DType](t *testing.T) {
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
			assertDenseDims(t, 1, len(tc.d), r.(*Dense[T]))
			assert.Equal(t, tc.d, Data[T](r))
		})
	}
}

func TestDense_ExtractColumn(t *testing.T) {
	t.Run("float32", testDenseExtractColumn[float32])
	t.Run("float64", testDenseExtractColumn[float64])
}

func testDenseExtractColumn[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.d, Data[T](c))
		})
	}
}

func TestDense_View(t *testing.T) {
	t.Run("float32", testDenseView[float32])
	t.Run("float64", testDenseView[float64])
}

func testDenseView[T float.DType](t *testing.T) {
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
		d.SetScalar(0, 0, float.Interface(T(42))) // modifying d must modify v too
		assert.Equal(t, float.Interface(T(42)), v.ScalarAt(0, 0))
		v.SetScalar(0, 0, float.Interface(T(2))) // modifying v must modify d too
		assert.Equal(t, float.Interface(T(2)), d.ScalarAt(0, 0))
	})
}

func TestDense_Slice(t *testing.T) {
	t.Run("float32", testDenseSlice[float32])
	t.Run("float64", testDenseSlice[float64])
}

func testDenseSlice[T float.DType](t *testing.T) {
	invalidTestCases := []struct {
		name                           string
		d                              *Dense[T]
		fromRow, fromCol, toRow, toCol int
	}{
		{"fromRow < 0", NewEmptyDense[T](2, 3), -1, 0, 2, 3},
		{"fromRow >= matrix rows", NewEmptyDense[T](2, 3), 2, 0, 2, 3},
		{"fromCol < 0", NewEmptyDense[T](2, 3), 0, -1, 2, 3},
		{"fromCol >= matrix cols", NewEmptyDense[T](2, 3), 0, 3, 2, 3},
		{"toRow < fromRow", NewEmptyDense[T](2, 3), 1, 0, 0, 3},
		{"toRow > matrix rows", NewEmptyDense[T](2, 3), 0, 0, 3, 3},
		{"toCol < fromCol", NewEmptyDense[T](2, 3), 0, 1, 2, 0},
		{"toCol > matrix cols", NewEmptyDense[T](2, 3), 0, 0, 2, 4},
		{"zero rows and cols", NewEmptyDense[T](0, 0), 0, 0, 0, 0},
		{"zero rows", NewEmptyDense[T](0, 2), 0, 0, 0, 2},
		{"zero cols", NewEmptyDense[T](2, 0), 0, 0, 2, 0},
	}

	for _, tc := range invalidTestCases {
		t.Run(tc.name, func(t *testing.T) {
			require.Panics(t, func() {
				tc.d.Slice(tc.fromRow, tc.fromCol, tc.toRow, tc.toCol)
			})
		})
	}

	d1x1 := NewDense[T](1, 1, []T{1})
	d4x4 := NewDense[T](4, 4, []T{
		11, 12, 13, 14,
		21, 22, 23, 24,
		31, 32, 33, 34,
		41, 42, 43, 44,
	})
	d1x3 := NewDense[T](1, 3, []T{1, 2, 3})
	d3x1 := NewDense[T](3, 1, []T{1, 2, 3})

	testCases := []struct {
		d                              *Dense[T]
		fromRow, fromCol, toRow, toCol int
		y                              []T
	}{
		{d1x1, 0, 0, 0, 0, []T{}},
		{d1x1, 0, 0, 1, 0, []T{}},
		{d1x1, 0, 0, 0, 1, []T{}},
		{d1x1, 0, 0, 1, 1, []T{1}},

		{d1x3, 0, 0, 0, 0, []T{}},
		{d1x3, 0, 0, 1, 1, []T{1}},
		{d1x3, 0, 1, 1, 2, []T{2}},
		{d1x3, 0, 2, 1, 3, []T{3}},
		{d1x3, 0, 0, 1, 2, []T{1, 2}},
		{d1x3, 0, 1, 1, 3, []T{2, 3}},
		{d1x3, 0, 0, 1, 3, []T{1, 2, 3}},

		{d3x1, 0, 0, 0, 0, []T{}},
		{d3x1, 0, 0, 1, 1, []T{1}},
		{d3x1, 1, 0, 2, 1, []T{2}},
		{d3x1, 2, 0, 3, 1, []T{3}},
		{d3x1, 0, 0, 2, 1, []T{1, 2}},
		{d3x1, 1, 0, 3, 1, []T{2, 3}},
		{d3x1, 0, 0, 3, 1, []T{1, 2, 3}},

		{d4x4, 0, 0, 0, 0, []T{}},
		{d4x4, 0, 0, 1, 1, []T{11}},

		{d4x4, 0, 0, 1, 2, []T{11, 12}},
		{d4x4, 0, 0, 1, 3, []T{11, 12, 13}},
		{d4x4, 0, 0, 1, 4, []T{11, 12, 13, 14}},
		{d4x4, 0, 1, 1, 4, []T{12, 13, 14}},
		{d4x4, 0, 2, 1, 4, []T{13, 14}},
		{d4x4, 0, 3, 1, 4, []T{14}},

		{d4x4, 0, 0, 2, 1, []T{11, 21}},
		{d4x4, 0, 0, 3, 1, []T{11, 21, 31}},
		{d4x4, 0, 0, 4, 1, []T{11, 21, 31, 41}},
		{d4x4, 1, 0, 4, 1, []T{21, 31, 41}},
		{d4x4, 2, 0, 4, 1, []T{31, 41}},
		{d4x4, 3, 0, 4, 1, []T{41}},

		{d4x4, 0, 1, 1, 3, []T{12, 13}},
		{d4x4, 1, 1, 2, 3, []T{22, 23}},
		{d4x4, 2, 1, 3, 3, []T{32, 33}},
		{d4x4, 3, 1, 4, 3, []T{42, 43}},

		{d4x4, 1, 0, 3, 1, []T{21, 31}},
		{d4x4, 1, 1, 3, 2, []T{22, 32}},
		{d4x4, 1, 2, 3, 3, []T{23, 33}},
		{d4x4, 1, 3, 3, 4, []T{24, 34}},

		{d4x4, 1, 1, 2, 2, []T{22}},
		{d4x4, 3, 3, 4, 4, []T{44}},

		{d4x4, 0, 0, 2, 2, []T{
			11, 12,
			21, 22,
		}},
		{d4x4, 0, 0, 2, 3, []T{
			11, 12, 13,
			21, 22, 23,
		}},
		{d4x4, 0, 0, 3, 2, []T{
			11, 12,
			21, 22,
			31, 32,
		}},
		{d4x4, 1, 1, 3, 3, []T{
			22, 23,
			32, 33,
		}},
		{d4x4, 2, 2, 4, 4, []T{
			33, 34,
			43, 44,
		}},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf(
			"%d x %d slice from (%d, %d) to (%d, %d)", tc.d.rows, tc.d.cols,
			tc.fromRow, tc.fromCol, tc.toRow, tc.toCol,
		)
		t.Run(name, func(t *testing.T) {
			y := tc.d.Slice(tc.fromRow, tc.fromCol, tc.toRow, tc.toCol)
			assertDenseDims(t, tc.toRow-tc.fromRow, tc.toCol-tc.fromCol, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Reshape(t *testing.T) {
	t.Run("float32", testDenseReshape[float32])
	t.Run("float64", testDenseReshape[float64])
}

func testDenseReshape[T float.DType](t *testing.T) {
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
		d.SetScalar(0, 0, float.Interface(T(42))) // modifying d must not modify r
		assert.Equal(t, float.Interface(T(0)), r.ScalarAt(0, 0))
	})
}

func TestDense_ReshapeInPlace(t *testing.T) {
	t.Run("float32", testDenseReshapeInPlace[float32])
	t.Run("float64", testDenseReshapeInPlace[float64])
}

func testDenseReshapeInPlace[T float.DType](t *testing.T) {
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

type flattenTestCase[T float.DType] struct {
	x *Dense[T]
	y []T
}

func flattenTestCases[T float.DType]() []flattenTestCase[T] {
	return []flattenTestCase[T]{
		{NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{1}), []T{1}},
		{NewDense[T](1, 2, []T{1, 2}), []T{1, 2}},
		{NewDense[T](2, 1, []T{1, 2}), []T{1, 2}},
		{
			NewDense[T](2, 2, []T{
				1, 2,
				3, 4,
			}),
			[]T{1, 2, 3, 4},
		},
		{
			NewDense[T](3, 4, []T{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			}),
			[]T{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		},
	}
}

func TestDense_Flatten(t *testing.T) {
	t.Run("float32", testDenseFlatten[float32])
	t.Run("float64", testDenseFlatten[float64])
}

func testDenseFlatten[T float.DType](t *testing.T) {
	for _, tc := range flattenTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d", tc.x.rows, tc.x.cols), func(t *testing.T) {
			y := tc.x.Flatten()
			assertDenseDims(t, 1, len(tc.y), y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_FlattenInPlace(t *testing.T) {
	t.Run("float32", testDenseFlattenInPlace[float32])
	t.Run("float64", testDenseFlattenInPlace[float64])
}

func testDenseFlattenInPlace[T float.DType](t *testing.T) {
	for _, tc := range flattenTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d", tc.x.rows, tc.x.cols), func(t *testing.T) {
			x2 := tc.x.FlattenInPlace()
			assert.Same(t, tc.x, x2)
			assertDenseDims(t, 1, len(tc.y), tc.x)
			assert.Equal(t, tc.y, tc.x.data)
		})
	}
}

func TestDense_ResizeVector(t *testing.T) {
	t.Run("float32", testDenseResizeVector[float32])
	t.Run("float64", testDenseResizeVector[float64])
}

func testDenseResizeVector[T float.DType](t *testing.T) {
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
		t.Run(fmt.Sprintf("column vector size %d resize %d", tc.size, tc.newSize), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.size, 1, func(r, _ int) T {
				return T(r + 1)
			})
			r := d.ResizeVector(tc.newSize)
			assert.Equal(t, tc.d, Data[T](r))
		})

		t.Run(fmt.Sprintf("row vector size %d resize %d", tc.size, tc.newSize), func(t *testing.T) {
			d := NewInitFuncDense[T](1, tc.size, func(_, c int) T {
				return T(c + 1)
			})
			r := d.ResizeVector(tc.newSize)
			assert.Equal(t, tc.d, Data[T](r))
		})
	}

	t.Run("data is copied - smaller size", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		r := d.ResizeVector(1)
		d.SetScalar(0, 0, float.Interface(T(42))) // modifying d must not modify r
		assert.Equal(t, float.Interface(T(0)), r.ScalarAt(0, 0))
	})

	t.Run("data is copied - bigger size", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		r := d.ResizeVector(3)
		d.SetScalar(0, 0, float.Interface(T(42))) // modifying d must not modify r
		assert.Equal(t, float.Interface(T(0)), r.ScalarAt(0, 0))
	})
}

func TestDense_T(t *testing.T) {
	t.Run("float32", testDenseT[float32])
	t.Run("float64", testDenseT[float64])
}

func testDenseT[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.d, Data[T](tr))
		})
	}
}

func TestDense_TransposeInPlace(t *testing.T) {
	t.Run("float32", testDenseTransposeInPlace[float32])
	t.Run("float64", testDenseTransposeInPlace[float64])
}

func testDenseTransposeInPlace[T float.DType](t *testing.T) {
	testCases := []struct {
		r int
		c int
		d []T
	}{
		// Each value is a 2-digit number having the format "<row><col>"
		{0, 0, []T{}},
		{0, 1, []T{}},
		{1, 0, []T{}},

		// Scalar
		{1, 1, []T{11}},

		// Vectors
		{1, 2, []T{11, 12}},
		{2, 1, []T{11, 21}},
		{1, 3, []T{11, 12, 13}},
		{3, 1, []T{11, 21, 31}},

		// Square matrix
		{2, 2, []T{
			11, 21,
			12, 22,
		}},
		{3, 3, []T{
			11, 21, 31,
			12, 22, 32,
			13, 23, 33,
		}},

		// Rectangular matrix
		{2, 3, []T{
			11, 21,
			12, 22,
			13, 23,
		}},
		{3, 2, []T{
			11, 21, 31,
			12, 22, 32,
		}},
		{3, 4, []T{
			11, 21, 31,
			12, 22, 32,
			13, 23, 33,
			14, 24, 34,
		}},
		{4, 3, []T{
			11, 21, 31, 41,
			12, 22, 32, 42,
			13, 23, 33, 43,
		}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.r, tc.c), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.r, tc.c, func(r int, c int) T {
				return T(c + 1 + (r+1)*10)
			})
			d2 := d.TransposeInPlace()
			assert.Same(t, d, d2)
			assertDenseDims(t, tc.c, tc.r, d)
			assert.Equal(t, tc.d, d.data)
		})
	}
}

type addTestCase[T float.DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func addTestCases[T float.DType]() []addTestCase[T] {
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

func testDenseAdd[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_AddInPlace(t *testing.T) {
	t.Run("float32", testDenseAddInPlace[float32])
	t.Run("float64", testDenseAddInPlace[float64])
}

func testDenseAddInPlace[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](tc.a))
		})
	}
}

type addScalarTestCase[T float.DType] struct {
	a *Dense[T]
	n float64
	y []T
}

func addScalarTestCases[T float.DType]() []addScalarTestCase[T] {
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

func testDenseAddScalar[T float.DType](t *testing.T) {
	for _, tc := range addScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			y := tc.a.AddScalar(tc.n)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_AddScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseAddScalarInPlace[float32])
	t.Run("float64", testDenseAddScalarInPlace[float64])
}

func testDenseAddScalarInPlace[T float.DType](t *testing.T) {
	for _, tc := range addScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			a2 := tc.a.AddScalarInPlace(tc.n)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, Data[T](tc.a))
		})
	}
}

type subTestCase[T float.DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func subTestCases[T float.DType]() []subTestCase[T] {
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

func testDenseSub[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_SubInPlace(t *testing.T) {
	t.Run("float32", testDenseSubInPlace[float32])
	t.Run("float64", testDenseSubInPlace[float64])
}

func testDenseSubInPlace[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](tc.a))
		})
	}
}

type subScalarTestCase[T float.DType] struct {
	a *Dense[T]
	n float64
	y []T
}

func subScalarTestCases[T float.DType]() []subScalarTestCase[T] {
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

func testDenseSubScalar[T float.DType](t *testing.T) {
	for _, tc := range subScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			y := tc.a.SubScalar(tc.n)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_SubScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseSubScalarInPlace[float32])
	t.Run("float64", testDenseSubScalarInPlace[float64])
}

func testDenseSubScalarInPlace[T float.DType](t *testing.T) {
	for _, tc := range subScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			a2 := tc.a.SubScalarInPlace(tc.n)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, Data[T](tc.a))
		})
	}
}

type prodTestCase[T float.DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func prodTestCases[T float.DType]() []prodTestCase[T] {
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

func testDenseProd[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_ProdInPlace(t *testing.T) {
	t.Run("float32", testDenseProdInPlace[float32])
	t.Run("float64", testDenseProdInPlace[float64])
}

func testDenseProdInPlace[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](tc.a))
		})
	}
}

type prodScalarTestCase[T float.DType] struct {
	a *Dense[T]
	n float64
	y []T
}

func prodScalarTestCases[T float.DType]() []prodScalarTestCase[T] {
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

func testDenseProdScalar[T float.DType](t *testing.T) {
	for _, tc := range prodScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			y := tc.a.ProdScalar(tc.n)
			assertDenseDims(t, tc.a.rows, tc.a.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_ProdScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseProdScalarInPlace[float32])
	t.Run("float64", testDenseProdScalarInPlace[float64])
}

func testDenseProdScalarInPlace[T float.DType](t *testing.T) {
	for _, tc := range prodScalarTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d, %g", tc.a.rows, tc.a.cols, tc.n), func(t *testing.T) {
			a2 := tc.a.ProdScalarInPlace(tc.n)
			assert.Same(t, tc.a, a2)
			assert.Equal(t, tc.y, Data[T](tc.a))
		})
	}
}

func TestDense_ProdMatrixScalarInPlace(t *testing.T) {
	t.Run("float32", testDenseProdMatrixScalarInPlace[float32])
	t.Run("float64", testDenseProdMatrixScalarInPlace[float64])
}

func testDenseProdMatrixScalarInPlace[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

type divTestCase[T float.DType] struct {
	a *Dense[T]
	b *Dense[T]
	y []T
}

func divTestCases[T float.DType]() []divTestCase[T] {
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

func testDenseDiv[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_DivInPlace(t *testing.T) {
	t.Run("float32", testDenseDivInPlace[float32])
	t.Run("float64", testDenseDivInPlace[float64])
}

func testDenseDivInPlace[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](tc.a))
		})
	}
}

func TestDense_Mul(t *testing.T) {
	t.Run("float32", testDenseMul[float32])
	t.Run("float64", testDenseMul[float64])
}

func testDenseMul[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_MulT(t *testing.T) {
	t.Run("float32", testDenseMulT[float32])
	t.Run("float64", testDenseMulT[float64])
}

func testDenseMulT[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_DotUnitary(t *testing.T) {
	t.Run("float32", testDenseDotUnitary[float32])
	t.Run("float64", testDenseDotUnitary[float64])
}

func testDenseDotUnitary[T float.DType](t *testing.T) {
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
			assertDenseDims(t, 1, 1, v.(*Dense[T]))
			assert.Equal(t, []T{tc.v}, Data[T](v))
		})
	}
}

func TestDense_ClipInPlace(t *testing.T) {
	t.Run("float32", testDenseClipInPlace[float32])
	t.Run("float64", testDenseClipInPlace[float64])
}

func testDenseClipInPlace[T float.DType](t *testing.T) {
	t.Run("max < min", func(t *testing.T) {
		d := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			d.ClipInPlace(2, 1)
		})
	})

	testCases := []struct {
		d        *Dense[T]
		min      float64
		max      float64
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

func testDenseMaximum[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Minimum(t *testing.T) {
	t.Run("float32", testDenseMinimum[float32])
	t.Run("float64", testDenseMinimum[float64])
}

func testDenseMinimum[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Abs(t *testing.T) {
	t.Run("float32", testDenseAbs[float32])
	t.Run("float64", testDenseAbs[float64])
}

func testDenseAbs[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Pow(t *testing.T) {
	t.Run("float32", testDensePow[float32])
	t.Run("float64", testDensePow[float64])
}

func testDensePow[T float.DType](t *testing.T) {
	testCases := []struct {
		d   *Dense[T]
		pow float64
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Sqrt(t *testing.T) {
	t.Run("float32", testDenseSqrt[float32])
	t.Run("float64", testDenseSqrt[float64])
}

func testDenseSqrt[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Sum(t *testing.T) {
	t.Run("float32", testDenseSum[float32])
	t.Run("float64", testDenseSum[float64])
}

func testDenseSum[T float.DType](t *testing.T) {
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
			assertDenseDims(t, 1, 1, y.(*Dense[T]))
			assert.Equal(t, []T{tc.y}, Data[T](y))
		})
	}
}

func TestDense_Max(t *testing.T) {
	t.Run("float32", testDenseMax[float32])
	t.Run("float64", testDenseMax[float64])
}

func testDenseMax[T float.DType](t *testing.T) {
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
			assertDenseDims(t, 1, 1, y.(*Dense[T]))
			assert.Equal(t, []T{tc.y}, Data[T](y))
		})
	}
}

func TestDense_Min(t *testing.T) {
	t.Run("float32", testDenseMin[float32])
	t.Run("float64", testDenseMin[float64])
}

func testDenseMin[T float.DType](t *testing.T) {
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
			assertDenseDims(t, 1, 1, y.(*Dense[T]))
			assert.Equal(t, []T{tc.y}, Data[T](y))
		})
	}
}

func TestDense_ArgMax(t *testing.T) {
	t.Run("float32", testDenseArgMax[float32])
	t.Run("float64", testDenseArgMax[float64])
}

func testDenseArgMax[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.ArgMax()
		})
	})

	t.Run("empty vector", func(t *testing.T) {
		d := NewEmptyVecDense[T](0)
		require.Panics(t, func() {
			d.ArgMax()
		})
	})

	testCases := []struct {
		d []T
		y int
	}{
		{[]T{0}, 0},
		{[]T{1}, 0},
		{[]T{-1}, 0},
		{[]T{3, 2}, 0},
		{[]T{2, 3}, 1},
		{[]T{1, -2, 3, -4}, 2},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("column vector %v", tc.d), func(t *testing.T) {
			d := NewDense[T](len(tc.d), 1, tc.d)
			y := d.ArgMax()
			assert.Equal(t, tc.y, y)
		})

		t.Run(fmt.Sprintf("row vector %v", tc.d), func(t *testing.T) {
			d := NewDense[T](1, len(tc.d), tc.d)
			y := d.ArgMax()
			assert.Equal(t, tc.y, y)
		})
	}
}

func TestDense_Softmax(t *testing.T) {
	t.Run("float32", testDenseSoftmax[float32])
	t.Run("float64", testDenseSoftmax[float64])
}

func testDenseSoftmax[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Softmax()
		})
	})

	testCases := []struct {
		x []T
		y []T
	}{
		{[]T{}, []T{}},
		{[]T{0}, []T{1}},
		{[]T{1}, []T{1}},
		{[]T{1, 2}, []T{0.26894142, 0.73105858}},
		{[]T{-1, 1}, []T{0.11920292, 0.88079708}},
		{[]T{0, 1, 2, 1}, []T{0.07232949, 0.19661193, 0.53444665, 0.19661193}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("column vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](len(tc.x), 1, tc.x)
			y := d.Softmax()
			assertDenseDims(t, len(tc.x), 1, y.(*Dense[T]))
			assert.InDeltaSlice(t, tc.y, y.Data(), 1e-7)
		})

		t.Run(fmt.Sprintf("row vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](1, len(tc.x), tc.x)
			y := d.Softmax()
			assertDenseDims(t, len(tc.x), 1, y.(*Dense[T]))
			assert.InDeltaSlice(t, tc.y, y.Data(), 1e-7)
		})
	}
}

func TestDense_CumSum(t *testing.T) {
	t.Run("float32", testDenseCumSum[float32])
	t.Run("float64", testDenseCumSum[float64])
}

func testDenseCumSum[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.CumSum()
		})
	})

	testCases := []struct {
		x []T
		y []T
	}{
		{[]T{}, []T{}},
		{[]T{0}, []T{0}},
		{[]T{1}, []T{1}},
		{[]T{-1}, []T{-1}},
		{[]T{1, 1}, []T{1, 2}},
		{[]T{1, 2}, []T{1, 3}},
		{[]T{1, -1}, []T{1, 0}},
		{[]T{1, 2, 3}, []T{1, 3, 6}},
		{[]T{1, -2, 3, -4}, []T{1, -1, 2, -2}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("column vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](len(tc.x), 1, tc.x)
			y := d.CumSum()
			assertDenseDims(t, len(tc.x), 1, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})

		t.Run(fmt.Sprintf("row vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](1, len(tc.x), tc.x)
			y := d.CumSum()
			assertDenseDims(t, len(tc.x), 1, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Range(t *testing.T) {
	t.Run("float32", testDenseRange[float32])
	t.Run("float64", testDenseRange[float64])
}

func testDenseRange[T float.DType](t *testing.T) {
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
		t.Run(fmt.Sprintf("column vector size %v range %d, %d", tc.size, tc.start, tc.end), func(t *testing.T) {
			d := NewInitFuncDense[T](tc.size, 1, func(r, _ int) T {
				return T(r + 1)
			})
			y := d.Range(tc.start, tc.end)
			assertDenseDims(t, len(tc.y), 1, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})

		t.Run(fmt.Sprintf("row vector size %v range %d, %d", tc.size, tc.start, tc.end), func(t *testing.T) {
			d := NewInitFuncDense[T](1, tc.size, func(_, c int) T {
				return T(c + 1)
			})
			y := d.Range(tc.start, tc.end)
			assertDenseDims(t, len(tc.y), 1, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_SplitV(t *testing.T) {
	t.Run("float32", testDenseSplitV[float32])
	t.Run("float64", testDenseSplitV[float64])
}

func testDenseSplitV[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.SplitV(1)
		})
	})

	t.Run("negative size", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.SplitV(-1)
		})
	})

	t.Run("empty sizes", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		y := d.SplitV()
		assert.Nil(t, y)
	})

	t.Run("sizes out of bound", func(t *testing.T) {
		d := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			fmt.Println(d.SplitV(1, 1, 1))
		})
	})

	testCases := []struct {
		x     []T
		sizes []int
		y     [][]T
	}{
		{[]T{}, []int{0}, [][]T{{}}},
		{[]T{}, []int{0, 0}, [][]T{{}, {}}},

		{[]T{1}, []int{0}, [][]T{{}}},
		{[]T{1}, []int{0, 0}, [][]T{{}, {}}},

		{[]T{1}, []int{1}, [][]T{{1}}},
		{[]T{1}, []int{0, 1}, [][]T{{}, {1}}},
		{[]T{1}, []int{1, 0}, [][]T{{1}, {}}},

		{[]T{1, 2, 3}, []int{1}, [][]T{{1}}},
		{[]T{1, 2, 3}, []int{1, 1}, [][]T{{1}, {2}}},
		{[]T{1, 2, 3}, []int{1, 2}, [][]T{{1}, {2, 3}}},
		{[]T{1, 2, 3}, []int{1, 1, 1}, [][]T{{1}, {2}, {3}}},
		{[]T{1, 2, 3}, []int{2}, [][]T{{1, 2}}},
		{[]T{1, 2, 3}, []int{2, 1}, [][]T{{1, 2}, {3}}},
		{[]T{1, 2, 3}, []int{3}, [][]T{{1, 2, 3}}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("column vector %v sizes %v", tc.x, tc.sizes), func(t *testing.T) {
			d := NewDense[T](len(tc.x), 1, tc.x)
			y := d.SplitV(tc.sizes...)
			require.Len(t, y, len(tc.y))
			for i, v := range y {
				expectedData := tc.y[i]
				assertDenseDims(t, len(expectedData), 1, v.(*Dense[T]))
				assert.Equal(t, expectedData, Data[T](v))
			}
		})

		t.Run(fmt.Sprintf("row vector %v sizes %v", tc.x, tc.sizes), func(t *testing.T) {
			d := NewDense[T](1, len(tc.x), tc.x)
			y := d.SplitV(tc.sizes...)
			require.Len(t, y, len(tc.y))
			for i, v := range y {
				expectedData := tc.y[i]
				assertDenseDims(t, len(expectedData), 1, v.(*Dense[T]))
				assert.Equal(t, expectedData, Data[T](v))
			}
		})
	}
}

func TestDense_Augment(t *testing.T) {
	t.Run("float32", testDenseAugment[float32])
	t.Run("float64", testDenseAugment[float64])
}

func testDenseAugment[T float.DType](t *testing.T) {
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
			require.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_SwapInPlace(t *testing.T) {
	t.Run("float32", testDenseSwapInPlace[float32])
	t.Run("float64", testDenseSwapInPlace[float64])
}

func testDenseSwapInPlace[T float.DType](t *testing.T) {
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

func testDensePadRows[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_PadColumns(t *testing.T) {
	t.Run("float32", testDensePadColumns[float32])
	t.Run("float64", testDensePadColumns[float64])
}

func testDensePadColumns[T float.DType](t *testing.T) {
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
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_AppendRows(t *testing.T) {
	t.Run("float32", testDenseAppendRows[float32])
	t.Run("float64", testDenseAppendRows[float64])
}

func testDenseAppendRows[T float.DType](t *testing.T) {
	t.Run("non vector value", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		m := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			d.AppendRows(m)
		})
	})

	t.Run("vector of incompatible size", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		v := NewEmptyVecDense[T](2)
		require.Panics(t, func() {
			d.AppendRows(v)
		})
	})

	testCases := []struct {
		d  *Dense[T]
		vs [][]T
		y  []T
	}{
		{NewEmptyDense[T](0, 0), [][]T{}, []T{}},
		{NewEmptyDense[T](0, 1), [][]T{}, []T{}},
		{NewEmptyDense[T](0, 1), [][]T{{1}}, []T{1}},
		{
			NewDense[T](1, 1, []T{1}),
			[][]T{{2}},
			[]T{1, 2},
		},
		{
			NewEmptyDense[T](0, 3),
			[][]T{
				{1, 2, 3},
				{4, 5, 6},
			},
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
			[][]T{},
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
			[][]T{
				{7, 8, 9},
			},
			[]T{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
		},
		{
			NewDense[T](1, 3, []T{
				1, 2, 3,
			}),
			[][]T{
				{4, 5, 6},
				{7, 8, 9},
			},
			[]T{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("append %d column vectors to %d x %d matrix", len(tc.vs), tc.d.rows, tc.d.cols), func(t *testing.T) {
			vs := make([]Matrix, len(tc.vs))
			for i, v := range tc.vs {
				vs[i] = NewDense[T](len(v), 1, v)
			}
			y := tc.d.AppendRows(vs...)
			assertDenseDims(t, tc.d.rows+len(tc.vs), tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})

		t.Run(fmt.Sprintf("append %d row vectors to %d x %d matrix", len(tc.vs), tc.d.rows, tc.d.cols), func(t *testing.T) {
			vs := make([]Matrix, len(tc.vs))
			for i, v := range tc.vs {
				vs[i] = NewDense[T](1, len(v), v)
			}
			y := tc.d.AppendRows(vs...)
			assertDenseDims(t, tc.d.rows+len(tc.vs), tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_Norm(t *testing.T) {
	t.Run("float32", testDenseNorm[float32])
	t.Run("float64", testDenseNorm[float64])
}

func testDenseNorm[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Norm(2)
		})
	})

	testCases := []struct {
		x   []T
		pow float64
		y   T
	}{
		{[]T{}, 2, 0},
		{[]T{1}, 2, 1},
		{[]T{1, 2}, 2, 2.23607},
		{[]T{1, 2}, 3, 2.08008},
		{[]T{1, 2, 3}, 2, 3.74166},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("column vector %v norm pow %g", tc.x, tc.pow), func(t *testing.T) {
			d := NewDense[T](len(tc.x), 1, tc.x)
			y := d.Norm(tc.pow)
			assertDenseDims(t, 1, 1, y.(*Dense[T]))
			assert.InDeltaSlice(t, []T{tc.y}, y.Data(), 1.0e-04)
		})

		t.Run(fmt.Sprintf("row vector %v norm pow %g", tc.x, tc.pow), func(t *testing.T) {
			d := NewDense[T](1, len(tc.x), tc.x)
			y := d.Norm(tc.pow)
			assertDenseDims(t, 1, 1, y.(*Dense[T]))
			assert.InDeltaSlice(t, []T{tc.y}, y.Data(), 1.0e-04)
		})
	}
}

func TestDense_Normalize2(t *testing.T) {
	t.Run("float32", testDenseNormalize2[float32])
	t.Run("float64", testDenseNormalize2[float64])
}

func testDenseNormalize2[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Normalize2()
		})
	})

	testCases := []struct {
		x []T
		y []T
	}{
		{[]T{0}, []T{0}},
		{[]T{1, 2, 3}, []T{0.267261, 0.534522, 0.801784}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("column vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](len(tc.x), 1, tc.x)
			y := d.Normalize2()
			assertDenseDims(t, len(tc.y), 1, y.(*Dense[T]))
			assert.InDeltaSlice(t, tc.y, y.Data(), 1.0e-06)
		})

		t.Run(fmt.Sprintf("row vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](1, len(tc.x), tc.x)
			y := d.Normalize2()
			assertDenseDims(t, 1, len(tc.y), y.(*Dense[T]))
			assert.InDeltaSlice(t, tc.y, y.Data(), 1.0e-06)
		})
	}
}

func TestDense_Pivoting(t *testing.T) {
	t.Run("float32", testDensePivoting[float32])
	t.Run("float64", testDensePivoting[float64])
}

func testDensePivoting[T float.DType](t *testing.T) {
	t.Run("non square matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Pivoting(1)
		})
	})

	t.Run("negative row", func(t *testing.T) {
		d := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			d.Pivoting(-1)
		})
	})

	t.Run("row out of upper bound", func(t *testing.T) {
		d := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			d.Pivoting(2)
		})
	})

	t.Run("case without swapping", func(t *testing.T) {
		d := NewDense(4, 4, []T{
			11, 9, 24, 2,
			1, 5, 2, 6,
			3, 17, 18, 1,
			2, 5, 7, 1,
		})

		p, swap, positions := d.Pivoting(0)
		assertDenseDims(t, 4, 4, p.(*Dense[T]))
		assert.Equal(t, []T{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, Data[T](p))
		assert.False(t, swap)
		assert.Equal(t, [2]int{0, 0}, positions)
	})

	t.Run("case with swapping", func(t *testing.T) {
		d := NewDense(4, 4, []T{
			11, 9, 24, 2,
			1, 5, 2, 6,
			3, 17, 7, 1,
			2, 5, 18, 1,
		})

		p, swap, positions := d.Pivoting(0)
		assertDenseDims(t, 4, 4, p.(*Dense[T]))
		assert.Equal(t, []T{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, Data[T](p))
		assert.False(t, swap)
		assert.Equal(t, [2]int{0, 0}, positions)

		p, swap, positions = d.Pivoting(2)
		assertDenseDims(t, 4, 4, p.(*Dense[T]))
		assert.Equal(t, []T{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
			0, 0, 1, 0,
		}, Data[T](p))
		assert.True(t, swap)
		assert.Equal(t, [2]int{3, 2}, positions)

		p, swap, positions = d.Pivoting(1)
		assertDenseDims(t, 4, 4, p.(*Dense[T]))
		assert.Equal(t, []T{
			1, 0, 0, 0,
			0, 0, 1, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
		}, Data[T](p))
		assert.True(t, swap)
		assert.Equal(t, [2]int{2, 1}, positions)
	})
}

func TestDense_LU(t *testing.T) {
	t.Run("float32", testDenseLU[float32])
	t.Run("float64", testDenseLU[float64])
}

func testDenseLU[T float.DType](t *testing.T) {
	t.Run("non square matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.LU()
		})
	})

	testCases := []struct {
		d *Dense[T]
		l []T
		u []T
		p []T
	}{
		{NewEmptyDense[T](0, 0), []T{}, []T{}, []T{}},
		{
			NewDense[T](3, 3, []T{
				3, 3, 0,
				7, -5, -1,
				2, 8, 3,
			}),
			[]T{
				1, 0, 0,
				0.285714, 1, 0,
				0.428571, 0.54545, 1,
			},
			[]T{
				7, -5, -1,
				0, 9.42857, 3.28571,
				0, 0, -1.363636,
			},
			[]T{
				0, 1, 0,
				0, 0, 1,
				1, 0, 0,
			},
		},
		{
			NewDense[T](4, 4, []T{
				11, 9, 24, 2,
				1, 5, 2, 6,
				3, 17, 18, 1,
				2, 5, 7, 1,
			}),
			[]T{
				1, 0, 0, 0,
				0.27273, 1, 0, 0,
				0.09091, 0.2875, 1, 0,
				0.18182, 0.23125, 0.0036, 1,
			},
			[]T{
				11, 9, 24, 2,
				0, 14.54545, 11.45455, 0.45455,
				0, 0, -3.475, 5.6875,
				0, 0, 0, 0.51079,
			},
			[]T{
				1, 0, 0, 0,
				0, 0, 1, 0,
				0, 1, 0, 0,
				0, 0, 0, 1,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			l, u, p := tc.d.LU()

			assertDenseDims(t, tc.d.rows, tc.d.cols, l.(*Dense[T]))
			assert.InDeltaSlice(t, tc.l, l.Data(), 1.0e-04)

			assertDenseDims(t, tc.d.rows, tc.d.cols, u.(*Dense[T]))
			assert.InDeltaSlice(t, tc.u, u.Data(), 1.0e-04)

			assertDenseDims(t, tc.d.rows, tc.d.cols, p.(*Dense[T]))
			assert.InDeltaSlice(t, tc.p, p.Data(), 1.0e-04)
		})
	}
}

func TestDense_Inverse(t *testing.T) {
	t.Run("float32", testDenseInverse[float32])
	t.Run("float64", testDenseInverse[float64])
}

func testDenseInverse[T float.DType](t *testing.T) {
	t.Run("non square matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.Inverse()
		})
	})

	testCases := []struct {
		d *Dense[T]
		y []T
	}{
		{NewEmptyDense[T](0, 0), []T{}},
		{NewDense[T](1, 1, []T{1}), []T{1}},
		{
			NewDense[T](3, 3, []T{
				1, 2, 3,
				0, 1, 4,
				5, 6, 0,
			}),
			[]T{
				-24, 18, 5,
				20, -15, -4,
				-5, 4, 1,
			},
		},
		{
			NewDense[T](4, 4, []T{
				1, 1, 1, -1,
				1, 1, -1, 1,
				1, -1, 1, 1,
				-1, 1, 1, 1,
			}),
			[]T{
				0.25, 0.25, 0.25, -0.25,
				0.25, 0.25, -0.25, 0.25,
				0.25, -0.25, 0.25, 0.25,
				-0.25, 0.25, 0.25, 0.25,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Inverse()
			assertDenseDims(t, tc.d.rows, tc.d.cols, y.(*Dense[T]))
			assert.InDeltaSlice(t, tc.y, y.Data(), 1.0e-04)
		})
	}
}

type applyTestCase[T float.DType] struct {
	d *Dense[T]
	y []T
}

func applyTestCases[T float.DType]() []applyTestCase[T] {
	return []applyTestCase[T]{
		// Each transoformed value is a 3-digit number having the
		// format "<n><row><col>"
		{NewEmptyDense[T](0, 0), []T{}},
		{NewEmptyDense[T](0, 1), []T{}},
		{NewEmptyDense[T](1, 0), []T{}},
		{NewDense[T](1, 1, []T{2}), []T{211}},
		{NewDense[T](1, 2, []T{2, 3}), []T{211, 312}},
		{NewDense[T](2, 1, []T{2, 3}), []T{211, 321}},
		{
			NewDense[T](2, 2, []T{
				1, 2,
				3, 4,
			}),
			[]T{
				111, 212,
				321, 422,
			},
		},
	}
}

func TestDense_Apply(t *testing.T) {
	t.Run("float32", testDenseApply[float32])
	t.Run("float64", testDenseApply[float64])
}

func testDenseApply[T float.DType](t *testing.T) {
	for _, tc := range applyTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.Apply(func(r, c int, v float64) float64 {
				return float64(c+1+(r+1)*10) + v*100
			})
			assertDenseDims(t, tc.d.rows, tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_ApplyInPlace(t *testing.T) {
	t.Run("float32", testDenseApplyInPlace[float32])
	t.Run("float64", testDenseApplyInPlace[float64])
}

func testDenseApplyInPlace[T float.DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			a.ApplyInPlace(func(r, c int, v float64) float64 { return 1 }, b)
		})
	})

	for _, tc := range applyTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			// start with a "dirty" matrix to ensure it's correctly overwritten
			// and initial data is irrelevant
			y := tc.d.OnesLike()
			y2 := y.ApplyInPlace(func(r, c int, v float64) float64 {
				return float64(c+1+(r+1)*10) + v*100
			}, tc.d)
			assert.Same(t, y, y2)
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_ApplyWithAlpha(t *testing.T) {
	t.Run("float32", testDenseApplyWithAlpha[float32])
	t.Run("float64", testDenseApplyWithAlpha[float64])
}

func testDenseApplyWithAlpha[T float.DType](t *testing.T) {
	inAlpha := []float64{1, 2, 3}
	for _, tc := range applyTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			y := tc.d.ApplyWithAlpha(func(r, c int, v float64, alpha ...float64) float64 {
				assert.Equal(t, inAlpha, alpha)
				return float64(c+1+(r+1)*10) + v*100
			}, inAlpha...)
			assertDenseDims(t, tc.d.rows, tc.d.cols, y.(*Dense[T]))
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_ApplyWithAlphaInPlace(t *testing.T) {
	t.Run("float32", testDenseApplyWithAlphaInPlace[float32])
	t.Run("float64", testDenseApplyWithAlphaInPlace[float64])
}

func testDenseApplyWithAlphaInPlace[T float.DType](t *testing.T) {
	inAlpha := []float64{1, 2, 3}

	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			a.ApplyWithAlphaInPlace(
				func(r, c int, v float64, alpha ...float64) float64 { return 1 },
				b, inAlpha...,
			)
		})
	})

	for _, tc := range applyTestCases[T]() {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			// start with a "dirty" matrix to ensure it's correctly overwritten
			// and initial data is irrelevant
			y := tc.d.OnesLike()
			y2 := y.ApplyWithAlphaInPlace(func(r, c int, v float64, alpha ...float64) float64 {
				assert.Equal(t, inAlpha, alpha)
				return float64(c+1+(r+1)*10) + v*100
			}, tc.d, inAlpha...)
			assert.Same(t, y, y2)
			assert.Equal(t, tc.y, Data[T](y))
		})
	}
}

func TestDense_DoNonZero(t *testing.T) {
	t.Run("float32", testDenseDoNonZero[float32])
	t.Run("float64", testDenseDoNonZero[float64])
}

type doNonZeroVisit struct {
	r int
	c int
	v float64
}

func testDenseDoNonZero[T float.DType](t *testing.T) {
	testCases := []struct {
		d      *Dense[T]
		visits []doNonZeroVisit
	}{
		{NewEmptyDense[T](0, 0), []doNonZeroVisit{}},
		{NewEmptyDense[T](0, 1), []doNonZeroVisit{}},
		{NewEmptyDense[T](1, 0), []doNonZeroVisit{}},
		{NewEmptyDense[T](2, 2), []doNonZeroVisit{}},
		{NewDense[T](1, 1, []T{0}), []doNonZeroVisit{}},
		{
			NewDense[T](1, 1, []T{1}),
			[]doNonZeroVisit{
				{0, 0, 1},
			},
		},
		{
			NewDense[T](1, 2, []T{0, 1}),
			[]doNonZeroVisit{
				{0, 1, 1},
			},
		},
		{
			NewDense[T](2, 1, []T{0, 1}),
			[]doNonZeroVisit{
				{1, 0, 1},
			},
		},
		{
			NewDense[T](2, 2, []T{
				1, 2,
				3, 4,
			}),
			[]doNonZeroVisit{
				{0, 0, 1},
				{0, 1, 2},
				{1, 0, 3},
				{1, 1, 4},
			},
		},
		{
			NewDense[T](2, 2, []T{
				1, 0,
				0, 2,
			}),
			[]doNonZeroVisit{
				{0, 0, 1},
				{1, 1, 2},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d data %v", tc.d.rows, tc.d.cols, tc.d.data), func(t *testing.T) {
			visits := []doNonZeroVisit{}
			tc.d.DoNonZero(func(r, c int, v float64) {
				visits = append(visits, doNonZeroVisit{r, c, v})
			})
			assert.Equal(t, tc.visits, visits)
		})
	}
}

func TestDense_DoVecNonZero(t *testing.T) {
	t.Run("float32", testDenseDoVecNonZero[float32])
	t.Run("float64", testDenseDoVecNonZero[float64])
}

type doVecNonZeroVisit struct {
	i int
	v float64
}

func testDenseDoVecNonZero[T float.DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			d.DoVecNonZero(func(i int, v float64) {})
		})
	})

	testCases := []struct {
		x      []T
		visits []doVecNonZeroVisit
	}{
		{[]T{}, []doVecNonZeroVisit{}},
		{[]T{0}, []doVecNonZeroVisit{}},
		{[]T{0, 0}, []doVecNonZeroVisit{}},
		{
			[]T{1},
			[]doVecNonZeroVisit{
				{0, 1},
			},
		},
		{
			[]T{0, 1},
			[]doVecNonZeroVisit{
				{1, 1},
			},
		},
		{
			[]T{1, 2, 3},
			[]doVecNonZeroVisit{
				{0, 1},
				{1, 2},
				{2, 3},
			},
		},
		{
			[]T{1, 0, 2, 0, 3},
			[]doVecNonZeroVisit{
				{0, 1},
				{2, 2},
				{4, 3},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("column vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](len(tc.x), 1, tc.x)
			visits := []doVecNonZeroVisit{}
			d.DoVecNonZero(func(i int, v float64) {
				visits = append(visits, doVecNonZeroVisit{i, v})
			})
			assert.Equal(t, tc.visits, visits)
		})

		t.Run(fmt.Sprintf("row vector %v", tc.x), func(t *testing.T) {
			d := NewDense[T](1, len(tc.x), tc.x)
			visits := []doVecNonZeroVisit{}
			d.DoVecNonZero(func(i int, v float64) {
				visits = append(visits, doVecNonZeroVisit{i, v})
			})
			assert.Equal(t, tc.visits, visits)
		})
	}
}

func TestDense_Clone(t *testing.T) {
	t.Run("float32", testDenseClone[float32])
	t.Run("float64", testDenseClone[float64])
}

func testDenseClone[T float.DType](t *testing.T) {
	testCases := []*Dense[T]{
		NewEmptyDense[T](0, 0),
		NewEmptyDense[T](0, 1),
		NewEmptyDense[T](1, 0),
		NewDense[T](1, 1, []T{1}),
		NewDense[T](1, 2, []T{1, 2}),
		NewDense[T](2, 1, []T{1, 2}),
		NewDense[T](2, 2, []T{1, 2, 3, 4}),
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.rows, tc.cols), func(t *testing.T) {
			y := tc.Clone()
			assertDenseDims(t, tc.rows, tc.cols, y.(*Dense[T]))
			assert.Equal(t, tc.data, Data[T](y))
		})
	}

	t.Run("data is copied", func(t *testing.T) {
		d := NewDense(1, 1, []T{1})
		y := d.Clone()
		d.SetScalar(0, 0, float.Interface(T(42)))
		assert.Equal(t, float.Interface(T(1)), y.ScalarAt(0, 0))
	})
}

func TestDense_Copy(t *testing.T) {
	t.Run("float32", testDenseCopy[float32])
	t.Run("float64", testDenseCopy[float64])
}

func testDenseCopy[T float.DType](t *testing.T) {
	t.Run("incompatible dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 2)
		require.Panics(t, func() {
			a.Copy(b)
		})
	})

	testCases := []*Dense[T]{
		NewEmptyDense[T](0, 0),
		NewEmptyDense[T](0, 1),
		NewEmptyDense[T](1, 0),
		NewDense[T](1, 1, []T{1}),
		NewDense[T](1, 2, []T{1, 2}),
		NewDense[T](2, 1, []T{1, 2}),
		NewDense[T](2, 2, []T{1, 2, 3, 4}),
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.rows, tc.cols), func(t *testing.T) {
			// start with a "dirty" matrix to ensure it's correctly overwritten
			// and initial data is irrelevant
			y := tc.OnesLike()
			y.Copy(tc)
			assert.Equal(t, tc.data, Data[T](y))
		})
	}
}

func TestDense_String(t *testing.T) {
	t.Run("float32", testDenseString[float32])
	t.Run("float64", testDenseString[float64])
}

func testDenseString[T float.DType](t *testing.T) {
	prefix := "Matrix|Dense"
	switch any(T(0)).(type) {
	case float32:
		prefix += "[float32]"
	case float64:
		prefix += "[float64]"
	default:
		t.Fatalf("unexpected type %T", T(0))
	}

	testCases := []struct {
		d *Dense[T]
		s string
	}{
		{NewEmptyDense[T](0, 0), "(0×0)[]"},
		{NewEmptyDense[T](0, 1), "(0×1)[]"},
		{NewEmptyDense[T](1, 0), "(1×0)[]"},
		{NewScalar[T](42), "(1×1)[42]"},
		{NewVecDense([]T{1, 2, 3}), "(3×1)[1 2 3]"},
		{
			NewDense(2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			"(2×3)[1 2 3 4 5 6]",
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.d.rows, tc.d.cols), func(t *testing.T) {
			s := tc.d.String()
			assert.Equal(t, prefix+tc.s, s)
		})
	}
}

func assertDenseDims[T float.DType](t *testing.T, expectedRows, expectedCols int, d *Dense[T]) {
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
