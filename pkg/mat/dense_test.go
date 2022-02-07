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
	t.Run("Float", testNewDense[Float])
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
}

func TestNewVecDense(t *testing.T) {
	t.Run("float32", testNewVecDense[float32])
	t.Run("float64", testNewVecDense[float64])
	t.Run("Float", testNewVecDense[Float])
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
}

func TestNewScalar(t *testing.T) {
	t.Run("float32", testNewScalar[float32])
	t.Run("float64", testNewScalar[float64])
	t.Run("Float", testNewScalar[Float])
}

func testNewScalar[T DType](t *testing.T) {
	d := NewScalar(T(42))
	assertDenseDims(t, 1, 1, d)
	assert.Equal(t, []T{42}, d.Data())
}

func TestNewEmptyVecDense(t *testing.T) {
	t.Run("float32", testNewEmptyVecDense[float32])
	t.Run("float64", testNewEmptyVecDense[float64])
	t.Run("Float", testNewEmptyVecDense[Float])
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
	t.Run("Float", testNewEmptyDense[Float])
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
	t.Run("Float", testNewOneHotVecDense[Float])
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
	t.Run("Float", testNewInitDense[Float])
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

func TestNewInitVecDense(t *testing.T) {
	t.Run("float32", testNewInitVecDense[float32])
	t.Run("float64", testNewInitVecDense[float64])
	t.Run("Float", testNewInitVecDense[Float])
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
	t.Run("Float", testNewIdentityDense[Float])
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
	t.Run("Float", testDenseSetData[Float])
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

	t.Run("data is copied correctly", func(t *testing.T) {
		d := NewEmptyDense[T](2, 3)
		v := []T{1, 2, 3, 7, 8, 9}
		d.SetData(v)
		assert.Equal(t, v, d.data)
	})
}

func TestDense_IsVector(t *testing.T) {
	t.Run("float32", testDenseIsVector[float32])
	t.Run("float64", testDenseIsVector[float64])
	t.Run("Float", testDenseIsVector[Float])
}

func testDenseIsVector[T DType](t *testing.T) {
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
			require.Equal(t, tc.b, d.IsVector())
		})
	}
}

func TestDense_IsScalar(t *testing.T) {
	t.Run("float32", testDenseIsScalar[float32])
	t.Run("float64", testDenseIsScalar[float64])
	t.Run("Float", testDenseIsScalar[Float])
}

func testDenseIsScalar[T DType](t *testing.T) {
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
			require.Equal(t, tc.b, d.IsScalar())
		})
	}
}

func TestDense_SameDims(t *testing.T) {
	t.Run("float32", testDenseSameDims[float32])
	t.Run("float64", testDenseSameDims[float64])
	t.Run("Float", testDenseSameDims[Float])
}

func testDenseSameDims[T DType](t *testing.T) {
	t.Run("different dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](3, 2)
		assert.False(t, a.SameDims(b))
		assert.False(t, b.SameDims(a))
	})

	t.Run("same dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 3)
		assert.True(t, a.SameDims(b))
		assert.True(t, b.SameDims(a))
	})
}

func TestDense_ZerosLike(t *testing.T) {
	t.Run("float32", testDenseZerosLike[float32])
	t.Run("float64", testDenseZerosLike[float64])
	t.Run("Float", testDenseZerosLike[Float])
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
	t.Run("Float", testDenseOnesLike[Float])
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
	t.Run("Float", testDenseScalar[Float])
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
	t.Run("Float", testDenseZeros[Float])
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

func TestDense_AddScalar(t *testing.T) {
	t.Run("float32", testDenseAddScalar[float32])
	t.Run("float64", testDenseAddScalar[float64])
	t.Run("Float", testDenseAddScalar[Float])
}

func testDenseAddScalar[T DType](t *testing.T) {
	// TODO: this is just a quick test; test corner cases...
	a := NewVecDense([]T{1, 2, 3, 0})
	b := a.AddScalar(10)
	assertSliceEqualApprox(t, []T{11, 12, 13, 10}, b.Data())
}

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

func assertSliceEqualApprox[T DType](t *testing.T, expected, actual []T) {
	t.Helper()
	assert.InDeltaSlice(t, expected, actual, 1.0e-04)
}
