// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"github.com/stretchr/testify/assert"
	"reflect"
	"testing"
)

func TestDense_AddScalar(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	b := a.AddScalar(1.0)
	assert.InDeltaSlice(t, []Float{1.1, 1.2, 1.3, 1.0}, b.Data(), 1.0e-6)
}

func TestDense_AddScalarInPlace(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	a.AddScalarInPlace(1.0)
	assert.InDeltaSlice(t, []Float{1.1, 1.2, 1.3, 1.0}, a.Data(), 1.0e-6)
}

func TestDense_SubScalar(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	b := a.SubScalar(2.0)
	assert.InDeltaSlice(t, []Float{-1.9, -1.8, -1.7, -2.0}, b.Data(), 1.0e-6)
}

func TestDense_SubScalarInPlace(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	a.SubScalarInPlace(2.0)
	assert.InDeltaSlice(t, []Float{-1.9, -1.8, -1.7, -2.0}, a.Data(), 1.0e-6)
}

func TestDense_Add(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		c := a.Add(b)
		assert.InDeltaSlice(t, []Float{0.5, 0.5, 0.8, 0.7}, c.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.Add(other) })
	})
}

func TestDense_AddInPlace(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		a.AddInPlace(b)

		assert.InDeltaSlice(t, []Float{0.5, 0.5, 0.8, 0.7}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.AddInPlace(other) })
	})
}

func TestDense_Sub(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		c := a.Sub(b)

		assert.InDeltaSlice(t, []Float{-0.3, -0.1, -0.2, -0.7}, c.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.Sub(other) })
	})
}

func TestDense_SubInPlace(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		a.SubInPlace(b)

		assert.InDeltaSlice(t, []Float{-0.3, -0.1, -0.2, -0.7}, a.Data(), 1.0e-6)
	})

	t.Run("it works with another Sparse matrix", func(t *testing.T) {
		d := NewDense(2, 3, []Float{
			10, 20, 30,
			40, 50, 60,
		})
		other := NewSparse(2, 3, []Float{
			1, 2, 3,
			4, 5, 6,
		})
		d.SubInPlace(other)
		expected := []Float{
			9, 18, 27,
			36, 45, 54,
		}
		assert.Equal(t, expected, d.Data())
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.SubInPlace(other) })
	})
}

func TestDense_ProdScalar(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	b := a.ProdScalar(2.0)
	assert.InDeltaSlice(t, []Float{0.2, 0.4, 0.6, 0.0}, b.Data(), 1.0e-6)
}

func TestDense_ProdScalarInPlace(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	a.ProdScalarInPlace(2.0)
	assert.InDeltaSlice(t, []Float{0.2, 0.4, 0.6, 0.0}, a.Data(), 1.0e-6)
}

func TestDense_Prod(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		c := a.Prod(b)
		assert.InDeltaSlice(t, []Float{0.04, 0.06, 0.15, 0}, c.Data(), 1.0e-6)
	})

	t.Run("it works with empty matrices", func(t *testing.T) {
		d := NewEmptyDense(0, 0)
		other := NewEmptyDense(0, 0)
		assert.NotPanics(t, func() { d.Prod(other) })
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.Prod(other) })
	})
}

func TestDense_ProdInPlace(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		a.ProdInPlace(b)

		assert.InDeltaSlice(t, []Float{0.04, 0.06, 0.15, 0}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.ProdInPlace(other) })
	})
}

func TestDense_ProdMatrixScalarInPlace(t *testing.T) {
	a := NewVecDense([]Float{0.0, 0.0, 0.0, 0.0})
	b := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	a.ProdMatrixScalarInPlace(b, 2.0)

	assert.InDeltaSlice(t, []Float{0.2, 0.4, 0.6, 0.0}, a.Data(), 1.0e-6)
}

func TestDense_Div(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		c := a.Div(b)

		assert.InDeltaSlice(t, []Float{0.25, 0.6666666666, 0.6, 0.0}, c.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.Div(other) })
	})
}

func TestDense_DivInPlace(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		b := NewVecDense([]Float{0.4, 0.3, 0.5, 0.7})
		a.DivInPlace(b)

		assert.InDeltaSlice(t, []Float{0.25, 0.6666666666, 0.6, 0.0}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.DivInPlace(other) })
	})
}

func TestDense_Mul(t *testing.T) {
	t.Run("matrix x matrix", func(t *testing.T) {
		a := NewDense(3, 4, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		})
		b := NewDense(4, 3, []Float{
			0.2, 0.7, 0.5,
			0.0, 0.4, 0.5,
			-0.8, 0.7, -0.3,
			0.2, -0.0, -0.9,
		})
		c := a.Mul(b)

		assert.InDeltaSlice(t, []Float{
			-0.22, 0.36, 0.06,
			0.7, 0.06, 0.0,
			0.52, -0.59, 0.48,
		}, c.Data(), 1.0e-6)
	})

	t.Run("matrix x vector", func(t *testing.T) {
		a := NewDense(3, 4, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		})
		b := NewVecDense([]Float{-0.8, -0.9, -0.9, 1.0})
		c := a.Mul(b)

		assert.InDeltaSlice(t, []Float{-0.53, 0.47, 0.3}, c.Data(), 1.0e-6)
	})

	t.Run("it works with another Sparse matrix", func(t *testing.T) {
		d := NewDense(2, 3, []Float{
			1, 2, 3,
			4, 5, 6,
		})
		other := NewSparse(3, 2, []Float{
			10, 20,
			30, 40,
			50, 60,
		})
		expected := []Float{
			220, 280,
			490, 640,
		}
		result := d.Mul(other)
		assert.Equal(t, expected, result.Data())
	})

	t.Run("it panics with incompatible dimensions", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(2, 4)
		assert.Panics(t, func() { d.Mul(other) })
	})
}

func TestDense_MulT(t *testing.T) {
	t.Run("column vector x column vector", func(t *testing.T) {
		d := NewDense(3, 1, []Float{
			1,
			2,
			3,
		})
		other := NewDense(3, 1, []Float{
			10,
			30,
			50,
		})
		expected := []Float{220}
		result := d.MulT(other)
		assert.Equal(t, expected, result.Data())
	})

	t.Run("it panics if rows differ", func(t *testing.T) {
		d := NewEmptyDense(3, 1)
		other := NewEmptyDense(2, 1)
		assert.Panics(t, func() { d.MulT(other) })
	})

	t.Run("it panics if the other matrix is not a column vector", func(t *testing.T) {
		d := NewEmptyDense(3, 1)
		other := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.MulT(other) })
	})

	t.Run("it panics if the other is Sparse", func(t *testing.T) {
		d := NewEmptyDense(3, 1)
		other := NewEmptySparse(3, 1)
		assert.Panics(t, func() { d.MulT(other) })
	})
}

func TestDense_Pow(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
	b := a.Pow(3.0)

	assert.InDeltaSlice(t, []Float{0.001, 0.008, 0.027, 0.0}, b.Data(), 1.0e-6)
}

func TestNewDense(t *testing.T) {
	t.Run("matrix 3 x 4", func(t *testing.T) {
		a := NewDense(3, 4, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		})

		if a.Rows() != 3 || a.Columns() != 4 {
			t.Error("The rows and columns are not correct")
		}

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}, a.Data(), 1.0e-6)
	})

	t.Run("square matrix 4 x 4", func(t *testing.T) {
		a := NewDense(4, 4, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
			0.9, 0.6, -0.2, 0.0,
		})

		if a.Rows() != 4 || a.Columns() != 4 {
			t.Error("The rows and columns are not correct")
		}

		if a.IsVector() {
			t.Error("The matrix shouldn't be a vector")
		}

		if a.IsScalar() {
			t.Error("The matrix shouldn't be a scalar")
		}

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
			0.9, 0.6, -0.2, 0.0,
		}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if elements is nil", func(t *testing.T) {
		assert.Panics(t, func() { NewDense(0, 0, nil) })
	})

	t.Run("it panics with an invalid elements size", func(t *testing.T) {
		assert.Panics(t, func() { NewDense(2, 3, []Float{1}) })
	})
}

func TestNewVecDense(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})

		if a.Rows() != 4 || a.Columns() != 1 {
			t.Error("The rows and columns are not correct")
		}

		if !a.IsVector() {
			t.Error("The matrix should be a vector")
		}

		if a.IsScalar() {
			t.Error("The matrix shouldn't be a scalar")
		}

		assert.InDeltaSlice(t, []Float{0.1, 0.2, 0.3, 0.0}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if elements is nil", func(t *testing.T) {
		assert.Panics(t, func() { NewVecDense(nil) })
	})
}

func TestNewScalar(t *testing.T) {
	a := NewScalar(0.42)

	if a.Rows() != 1 || a.Columns() != 1 {
		t.Error("The rows and columns are not correct")
	}

	if !a.IsScalar() {
		t.Error("The matrix should be a scalar")
	}

	if a.Scalar() != 0.42 {
		t.Error("The scalar doesn't match the expected value")
	}

	assert.InDeltaSlice(t, []Float{0.42}, a.Data(), 1.0e-6)
}

func TestNewEmptyVecDense(t *testing.T) {
	a := NewEmptyVecDense(4)

	if a.Rows() != 4 || a.Columns() != 1 {
		t.Error("The rows and columns are not correct")
	}

	if !a.IsVector() {
		t.Error("The matrix should be a vector")
	}

	if a.IsScalar() {
		t.Error("The matrix shouldn't be a scalar")
	}

	assert.InDeltaSlice(t, []Float{0.0, 0.0, 0.0, 0.0}, a.Data(), 1.0e-6)
}

func TestNewEmptyDenseNXM(t *testing.T) {
	a := NewEmptyDense(3, 4)

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	assert.InDeltaSlice(t, []Float{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, a.Data(), 1.0e-6)
}

func TestDense_ZerosLike(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.ZerosLike()

	if b.Rows() != 3 || b.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	assert.InDeltaSlice(t, []Float{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, b.Data(), 1.0e-6)
}

func TestDense_OnesLike(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.OnesLike()

	if b.Rows() != 3 || b.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	assert.InDeltaSlice(t, []Float{
		1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0,
	}, b.Data(), 1.0e-6)
}

func TestDense_Zeros(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	a.Zeros()

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	assert.InDeltaSlice(t, []Float{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, a.Data(), 1.0e-6)
}

func TestOneHotVecDense(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := OneHotVecDense(10, 8)

		if a.Rows() != 10 || a.Columns() != 1 {
			t.Error("The rows and columns are not correct")
		}

		if !a.IsVector() {
			t.Error("The matrix should be a vector")
		}

		if a.IsScalar() {
			t.Error("The matrix shouldn't be a scalar")
		}

		assert.InDeltaSlice(t, []Float{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if oneAt >= size", func(t *testing.T) {
		assert.Panics(t, func() { OneHotVecDense(2, 2) })
	})
}

func TestNewInitDenseNXM(t *testing.T) {
	a := NewInitDense(3, 4, 0.42)

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	assert.InDeltaSlice(t, []Float{
		0.42, 0.42, 0.42, 0.42,
		0.42, 0.42, 0.42, 0.42,
		0.42, 0.42, 0.42, 0.42,
	}, a.Data(), 1.0e-6)
}

func TestNewInitVecDense(t *testing.T) {
	a := NewInitVecDense(4, 0.42)

	if a.Rows() != 4 || a.Columns() != 1 {
		t.Error("The rows and columns are not correct")
	}

	if !a.IsVector() {
		t.Error("The matrix should be a vector")
	}

	if a.IsScalar() {
		t.Error("The matrix shouldn't be a scalar")
	}

	assert.InDeltaSlice(t, []Float{0.42, 0.42, 0.42, 0.42}, a.Data(), 1.0e-6)
}

func TestDense_Reshape(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(3, 4, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		})

		b := a.Reshape(4, 3)

		if b.Rows() != 4 || b.Columns() != 3 {
			t.Error("The rows and columns are not correct")
		}

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3,
			0.0, 0.4, 0.5,
			-0.6, 0.7, -0.5,
			0.8, -0.8, -0.1,
		}, b.Data(), 1.0e-6)
	})

	t.Run("it panics with incompatible size", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		assert.Panics(t, func() { d.Reshape(1, 2) })
	})
}

func TestDense_T(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.T()

	if b.Rows() != 4 || b.Columns() != 3 {
		t.Error("The rows and columns are not correct")
	}

	assert.InDeltaSlice(t, []Float{
		0.1, 0.4, -0.5,
		0.2, 0.5, 0.8,
		0.3, -0.6, -0.8,
		0.0, 0.7, -0.1,
	}, b.Data(), 1.0e-6)
}

func TestDense_Clone(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.Clone()

	if b.Rows() != 3 || b.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	assert.InDeltaSlice(t, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	}, b.Data(), 1.0e-6)
}

func TestDense_Copy(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(3, 4, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		})

		b := NewEmptyDense(a.Dims())
		b.Copy(a)

		if b.Rows() != 3 || b.Columns() != 4 {
			t.Error("The rows and columns are not correct")
		}

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}, b.Data(), 1.0e-6)
	})

	t.Run("it panics if dimensions differ", func(t *testing.T) {
		a := NewEmptyDense(1, 2)
		b := NewEmptyDense(2, 1)
		assert.Panics(t, func() { a.Copy(b) })
	})

	t.Run("it panics if the other matrix is not dense", func(t *testing.T) {
		a := NewEmptyDense(1, 2)
		b := NewEmptySparse(1, 2)
		assert.Panics(t, func() { a.Copy(b) })
	})
}

func TestDense_SizeMatrix(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	if a.Size() != 12 {
		t.Error("The size is not correct")
	}
}

func TestDense_SizeVector(t *testing.T) {
	a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})

	if a.Size() != 4 {
		t.Error("The size is not correct")
	}
}

func TestDense_SizeScalar(t *testing.T) {
	a := NewVecDense([]Float{0.42})

	if a.Size() != 1 {
		t.Error("The size is not correct")
	}
}

func TestDense_Dims(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	if a.Size() != 12 {
		t.Error("The size is not correct")
	}

	if a.LastIndex() != 11 {
		t.Error("The last index is not correct")
	}

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	r, c := a.Dims()
	if r != 3 || c != 4 {
		t.Error("The dims are not correct")
	}
}

func TestDense_Clip(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	a.ClipInPlace(0.1, 0.3)

	assert.InDeltaSlice(t, []Float{
		0.1, 0.2, 0.3, 0.1,
		0.3, 0.3, 0.1, 0.3,
		0.1, 0.3, 0.1, 0.1,
	}, a.Data(), 1.0e-6)
}

func TestDense_Abs(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.Abs()

	assert.InDeltaSlice(t, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, 0.6, 0.7,
		0.5, 0.8, 0.8, 0.1,
	}, b.Data(), 1.0e-6)
}

func TestDense_MaxMinSum(t *testing.T) {
	a := NewDense(3, 4, []Float{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	max := a.Max()
	min := a.Min()
	sum := a.Sum()

	if max != 0.8 {
		t.Error("The max doesn't match the expected value")
	}

	if min != -0.8 {
		t.Error("The max doesn't match the expected value")
	}

	if sum != 1.0 {
		t.Error("The sum doesn't match the expected value")
	}
}

func TestDense_Identity(t *testing.T) {
	a := I(3)

	assert.InDeltaSlice(t, []Float{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
	}, a.Data(), 1.0e-6)
}

func TestDense_Pivoting(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 4, []Float{
			11, 9, 24, 2,
			1, 5, 2, 6,
			3, 17, 18, 1,
			2, 5, 7, 1,
		})
		n := NewDense(4, 4, []Float{
			11, 9, 24, 2,
			1, 5, 2, 6,
			3, 17, 7, 1,
			2, 5, 18, 1,
		})

		b, s, positions := a.Pivoting(0)

		assert.InDeltaSlice(t, []Float{
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0,
		}, b.Data(), 1.0e-6)
		if s {
			t.Error("The flag doesn't match the expected values")
		}
		if !(reflect.DeepEqual(positions, []int{0, 0})) {
			t.Error("The data doesn't match the expected values")
		}

		c, s, positions := n.Pivoting(2)

		assert.InDeltaSlice(t, []Float{
			1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0,
			0.0, 0.0, 1.0, 0.0,
		}, c.Data(), 1.0e-6)
		if !s {
			t.Error("The flag doesn't match the expected values")
		}
		if !(reflect.DeepEqual(positions, []int{3, 2})) {
			t.Error("The data doesn't match the expected values")
		}

		d, s, positions := a.Pivoting(1)

		assert.InDeltaSlice(t, []Float{
			1.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0,
		}, d.Data(), 1.0e-6)
		if !s {
			t.Error("The flag doesn't match the expected values")
		}
		if !(reflect.DeepEqual(positions, []int{2, 1})) {
			t.Error("The data doesn't match the expected values")
		}
	})

	t.Run("it panics if the matrix is not square", func(t *testing.T) {
		d := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.Pivoting(1) })
	})
}

func TestDense_LU(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(3, 3, []Float{
			3, 3, 0,
			7, -5, -1,
			2, 8, 3,
		})

		l, u, p := a.LU()

		assert.InDeltaSlice(t, []Float{
			1, 0, 0,
			0.285714, 1, 0,
			0.428571, 0.54545, 1,
		}, l.Data(), 1.0e-5)

		assert.InDeltaSlice(t, []Float{
			7, -5, -1,
			0, 9.42857, 3.28571,
			0, 0, -1.363636,
		}, u.Data(), 1.0e-5)

		assert.InDeltaSlice(t, []Float{
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0,
			1.0, 0.0, 0.0,
		}, p.Data(), 1.0e-6)

		b := NewDense(4, 4, []Float{
			11, 9, 24, 2,
			1, 5, 2, 6,
			3, 17, 18, 1,
			2, 5, 7, 1,
		})

		l2, u2, p2 := b.LU()

		assert.InDeltaSlice(t, []Float{
			1.0, 0.0, 0.0, 0.0,
			0.27273, 1.0, 0.0, 0.0,
			0.09091, 0.28750, 1.0, 0.0,
			0.18182, 0.23125, 0.00360, 1.0,
		}, l2.Data(), 1.0e-5)

		assert.InDeltaSlice(t, []Float{
			11.0000, 9.0, 24.0, 2.0,
			0.0, 14.54545, 11.45455, 0.45455,
			0.0, 0.0, -3.47500, 5.68750,
			0.0, 0.0, 0.0, 0.51079,
		}, u2.Data(), 1.0e-5)

		assert.InDeltaSlice(t, []Float{
			1.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0,
		}, p2.Data(), 1.0e-6)
	})

	t.Run("it panics if the matrix is not square", func(t *testing.T) {
		d := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.LU() })
	})
}

func TestDense_Inverse(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {

		a := NewDense(3, 3, []Float{
			1, 2, 3,
			0, 1, 4,
			5, 6, 0,
		})

		i := a.Inverse()

		assert.InDeltaSlice(t, []Float{
			-24, 18, 5,
			20, -15, -4,
			-5, 4, 1,
		}, i.Data(), 1.0e-5)

		b := NewDense(4, 4, []Float{
			0.3, 0.2, 0.6, -23,
			1, 1, -1, 5,
			6, -7.5, 3, 0,
			1, 0, 0, 0,
		})

		c := b.Inverse()

		assert.InDeltaSlice(t, []Float{
			0, 0, 0, 1,
			-0.19230769, -0.88461538, -0.25641025, 2.48076923,
			-0.48076923, -2.21153846, -0.30769230, 4.20192307,
			-0.05769230, -0.06538461, -0.01025641, 0.14423076,
		}, c.Data(), 1.0e-5)

		d := NewDense(4, 4, []Float{
			1, 1, 1, -1,
			1, 1, -1, 1,
			1, -1, 1, 1,
			-1, 1, 1, 1,
		})

		e := d.Inverse()

		assert.InDeltaSlice(t, []Float{
			0.25, 0.25, 0.25, -0.25,
			0.25, 0.25, -0.25, 0.25,
			0.25, -0.25, 0.25, 0.25,
			-0.25, 0.25, 0.25, 0.25,
		}, e.Data(), 1.0e-5)
	})

	t.Run("it panics if the matrix is not square", func(t *testing.T) {
		d := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.Inverse() })
	})
}

func TestDense_Augment(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(3, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
		})

		b := a.Augment()

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3, 1.0, 0.0, 0.0,
			0.4, 0.5, -0.6, 0.0, 1.0, 0.0,
			-0.5, 0.8, -0.8, 0.0, 0.0, 1.0,
		}, b.Data(), 1.0e-6)
	})

	t.Run("it panics if the matrix is not square", func(t *testing.T) {
		d := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.Augment() })
	})
}

func TestDense_SwapInPlace(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, -0.4,
		})
		a.SwapInPlace(3, 2)

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-3, -0.3, -0.4,
			-0.5, 0.8, -0.8,
		}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if it is a vector", func(t *testing.T) {
		d := NewEmptyVecDense(3)
		assert.Panics(t, func() { d.SwapInPlace(1, 2) })
	})

	t.Run("it panics if r1 >= rows", func(t *testing.T) {
		d := NewEmptyDense(3, 4)
		assert.Panics(t, func() { d.SwapInPlace(3, 2) })
	})

	t.Run("it panics if r2 >= rows", func(t *testing.T) {
		d := NewEmptyDense(3, 4)
		assert.Panics(t, func() { d.SwapInPlace(1, 3) })
	})
}

func TestDense_Maximum(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, -0.4,
		})
		b := NewDense(4, 3, []Float{
			0.2, 0.7, 0.5,
			0.0, 0.4, 0.5,
			-0.8, 0.7, -0.3,
			0.2, -0.0, -0.9,
		})
		c := a.Maximum(b)

		assert.InDeltaSlice(t, []Float{
			0.2, 0.7, 0.5,
			0.4, 0.5, 0.5,
			-0.5, 0.8, -0.3,
			0.2, -0.0, -0.4,
		}, c.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(3, 2)
		other := NewEmptyDense(2, 3)
		assert.Panics(t, func() { d.Maximum(other) })
	})
}

func TestDense_Minimum(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, -0.4,
		})
		b := NewDense(4, 3, []Float{
			0.2, 0.7, 0.5,
			0.0, 0.4, 0.5,
			-0.8, 0.7, -0.3,
			0.2, -0.0, -0.9,
		})
		c := a.Minimum(b)

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3,
			0.0, 0.4, -0.6,
			-0.8, 0.7, -0.8,
			-3, -0.3, -0.9,
		}, c.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(3, 2)
		other := NewEmptyDense(2, 3)
		assert.Panics(t, func() { d.Minimum(other) })
	})
}

func TestDense_ExtractRow(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, -0.4,
		})

		c := a.ExtractRow(2)

		assert.InDeltaSlice(t, []Float{-0.5, 0.8, -0.8}, c.Data(), 1.0e-6)
	})

	t.Run("it panics if i >= rows", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		assert.Panics(t, func() { d.ExtractRow(2) })
	})
}

func TestDense_ExtractColumn(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, -0.4,
		})

		c := a.ExtractColumn(2)

		assert.InDeltaSlice(t, []Float{0.3, -0.6, -0.8, -0.4}, c.Data(), 1.0e-6)
	})

	t.Run("it panics if i >= columns", func(t *testing.T) {
		d := NewEmptyDense(3, 2)
		assert.Panics(t, func() { d.ExtractColumn(2) })
	})
}

func TestDense_Range(t *testing.T) {
	a := NewDense(4, 3, []Float{
		0.1, 0.2, 0.3,
		0.4, 0.5, -0.6,
		-0.5, 0.8, -0.8,
		-3, -0.3, -0.4,
	})

	c := a.Range(3, 6)

	assert.InDeltaSlice(t, []Float{0.4, 0.5, -0.6}, c.Data(), 1.0e-6)
}

func TestDense_SplitV(t *testing.T) {
	a := NewDense(4, 3, []Float{
		0.1, 0.2, 0.3,
		0.4, 0.5, -0.6,
		-0.5, 0.8, -0.8,
		-3, -0.3, -0.4,
	})
	c := a.SplitV(3, 3, 3)

	assert.InDeltaSlice(t, []Float{0.1, 0.2, 0.3}, c[0].Data(), 1.0e-6)
	assert.InDeltaSlice(t, []Float{0.4, 0.5, -0.6}, c[1].Data(), 1.0e-6)
	assert.InDeltaSlice(t, []Float{-0.5, 0.8, -0.8}, c[2].Data(), 1.0e-6)
}

func TestDense_At(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, -0.4,
		})
		v := a.At(3, 2)

		if !(v == -0.4) {
			t.Error("The result doesn't match the expected values")
		}
	})

	t.Run("it panics if i >= rows", func(t *testing.T) {
		d := NewEmptyDense(2, 5)
		assert.Panics(t, func() { d.At(2, 4) })
	})

	t.Run("it panics if j >= cols", func(t *testing.T) {
		d := NewEmptyDense(5, 2)
		assert.Panics(t, func() { d.At(4, 2) })
	})
}

func TestDense_Set(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewDense(4, 3, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, -0.4,
		})
		a.Set(3, 2, 3.0)

		assert.InDeltaSlice(t, []Float{
			0.1, 0.2, 0.3,
			0.4, 0.5, -0.6,
			-0.5, 0.8, -0.8,
			-3, -0.3, 3.0,
		}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if i >= rows", func(t *testing.T) {
		d := NewEmptyDense(2, 5)
		assert.Panics(t, func() { d.Set(2, 4, 42) })
	})

	t.Run("it panics if j >= cols", func(t *testing.T) {
		d := NewEmptyDense(5, 2)
		assert.Panics(t, func() { d.Set(4, 2, 42) })
	})
}

func TestDense_AtVec(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		v := a.AtVec(2)

		if !(v == 0.3) {
			t.Error("The result doesn't match the expected values")
		}
	})

	t.Run("it panics if i >= rows", func(t *testing.T) {
		d := NewEmptyVecDense(3)
		assert.Panics(t, func() { d.AtVec(3) })
	})

	t.Run("it panics if it is not a vector", func(t *testing.T) {
		d := NewEmptyDense(5, 2)
		assert.Panics(t, func() { d.AtVec(3) })
	})
}

func TestDense_SetVec(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		a.SetVec(3, 3.0)

		assert.InDeltaSlice(t, []Float{0.1, 0.2, 0.3, 3.0}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if i >= rows", func(t *testing.T) {
		d := NewEmptyVecDense(3)
		assert.Panics(t, func() { d.SetVec(3, 42) })
	})

	t.Run("it panics if it is not a vector", func(t *testing.T) {
		d := NewEmptyDense(5, 2)
		assert.Panics(t, func() { d.SetVec(3, 42) })
	})
}

func TestDense_Sqrt(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{1.0, 2.0, 4.0, 0.0})
		c := a.Sqrt()

		assert.InDeltaSlice(t, []Float{1.0, 1.414213, 2.0, 0.0}, c.Data(), 1.0e-6)
	})

	t.Run("it works with empty matrices", func(t *testing.T) {
		d := NewEmptyDense(0, 0)
		result := d.Sqrt()
		assert.Equal(t, 0, result.Size())
	})
}

func TestDense_Apply(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		a.Apply(func(i, j int, v Float) Float {
			return -3.0 * (v / 2.0) // the equation is completely arbitrary
		}, a)

		assert.InDeltaSlice(t, []Float{-0.15, -0.3, -0.45, 0.0}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		f := func(i, j int, v Float) Float {
			t.Error("the callback function should never be invoked")
			return 0
		}
		assert.Panics(t, func() { d.Apply(f, other) })
	})

	t.Run("it works with empty matrices", func(t *testing.T) {
		d := NewEmptyDense(0, 0)
		other := NewEmptyDense(0, 0)
		f := func(i, j int, v Float) Float {
			t.Error("the callback function should never be invoked")
			return 0
		}
		assert.NotPanics(t, func() { d.Apply(f, other) })
	})

	t.Run("it works with another Sparse matrix", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewSparse(2, 3, []Float{
			1, 2, 3,
			4, 5, 6,
		})
		f := func(i, j int, v Float) Float {
			return Float((i+1)*10) + Float(j+1) + (v / 10)
		}
		d.Apply(f, other)
		expected := []Float{
			11.1, 12.2, 13.3,
			21.4, 22.5, 23.6,
		}
		assert.Equal(t, expected, d.Data())
	})
}

func TestDense_ApplyWithAlpha(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		a := NewVecDense([]Float{0.1, 0.2, 0.3, 0.0})
		a.ApplyWithAlpha(func(i, j int, v Float, alpha ...Float) Float {
			return -3.0*(v/2.0) + alpha[0] // the equation is completely arbitrary
		}, a, 2.0)

		assert.InDeltaSlice(t, []Float{1.85, 1.7, 1.55, 2.0}, a.Data(), 1.0e-6)
	})

	t.Run("it panics if matrices dimensions differ", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		other := NewEmptyDense(3, 2)
		f := func(i, j int, v Float, alpha ...Float) Float {
			t.Error("the callback function should never be invoked")
			return 0
		}
		assert.Panics(t, func() { d.ApplyWithAlpha(f, other, 0) })
	})
}

func TestDense_Stack(t *testing.T) {
	v1 := NewVecDense([]Float{0.1, 0.2, 0.3, 0.5})
	v2 := NewVecDense([]Float{0.4, 0.5, 0.6, 0.4})
	v3 := NewVecDense([]Float{0.8, 0.9, 0.7, 0.6})

	out := Stack(v1, v2, v3)

	assert.InDeltaSlice(t, []Float{0.1, 0.2, 0.3, 0.5, 0.4, 0.5, 0.6, 0.4, 0.8, 0.9, 0.7, 0.6}, out.Data(), 1.0e-6)
}

func TestDense_SetData(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		d := NewDense(2, 3, []Float{
			1, 2, 3,
			4, 5, 6,
		})
		d.SetData([]Float{10, 20, 30, 40, 50, 60})
		assert.Equal(t, d.Data(), []Float{10, 20, 30, 40, 50, 60})
	})

	t.Run("it panics with incompatible data dimension", func(t *testing.T) {
		d := NewDense(2, 3, []Float{
			1, 2, 3,
			4, 5, 6,
		})
		assert.Panics(t, func() { d.SetData([]Float{10, 20}) })
	})
}

func TestDense_View(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		d := NewDense(2, 3, []Float{
			1, 2, 3,
			4, 5, 6,
		})
		view := d.View(3, 2)
		actualRows, actualCols := view.Dims()
		assert.Equal(t, actualRows, 3)
		assert.Equal(t, actualCols, 2)
		assert.Equal(t, view.Data(), []Float{1, 2, 3, 4, 5, 6})
	})

	t.Run("it panics with incompatible dimensions", func(t *testing.T) {
		d := NewEmptyDense(2, 3)
		assert.Panics(t, func() { d.View(1, 2) })
	})
}

func TestDense_Scalar(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		s := NewScalar(42)
		assert.Equal(t, s.Scalar(), 42.0)
	})

	t.Run("it panics with a non-scalar matrix", func(t *testing.T) {
		d := NewEmptyDense(1, 2)
		assert.Panics(t, func() { d.Scalar() })
	})
}

func TestDense_DotUnitary(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		d := NewDense(1, 3, []Float{1, 2, 3})
		other := NewDense(1, 3, []Float{10, 20, 30})
		assert.Equal(t, d.DotUnitary(other), Float(140))
	})

	t.Run("it panics with incompatible dimensions", func(t *testing.T) {
		d := NewDense(1, 3, []Float{1, 2, 3})
		other := NewDense(1, 2, []Float{10, 20})
		assert.Panics(t, func() { d.DotUnitary(other) })
	})
}

func TestDense_Norm(t *testing.T) {
	d := NewVecDense([]Float{1, 2, 3})
	actual := d.Norm(2)
	assert.InDelta(t, 3.741657, actual, 1.0e-6)
}

func TestDense_Normalize2(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		d := NewVecDense([]Float{1, 2, 3})
		actual := d.Normalize2().Data()
		expected := []Float{0.267261, 0.534522, 0.801784}
		assert.InDeltaSlice(t, expected, actual, 1.0e-6)
	})

	t.Run("with norm = 0", func(t *testing.T) {
		d := NewVecDense([]Float{0})
		actual := d.Normalize2().Data()
		expected := []Float{0}
		assert.Equal(t, expected, actual)
	})
}

func TestDense_DoNonZero(t *testing.T) {
	t.Run("empty matrix", func(t *testing.T) {
		m := NewEmptyDense(0, 0)
		fn := func(i, j int, v Float) {
			t.Fatalf("unexpected call with %d, %d, %f", i, j, v)
		}
		m.DoNonZero(fn)
	})

	t.Run("simple case", func(t *testing.T) {
		m := NewDense(2, 3, []Float{
			10, 0, 20,
			0, 30, 0,
		})
		type visited struct {
			i, j int
			f    Float
		}
		actual := make([]visited, 0)
		fn := func(i, j int, v Float) {
			actual = append(actual, visited{i, j, v})
		}
		m.DoNonZero(fn)
		expected := []visited{
			{0, 0, 10},
			{0, 2, 20},
			{1, 1, 30},
		}
		assert.Equal(t, expected, actual)
	})
}

func TestDense_String(t *testing.T) {
	d := NewVecDense([]Float{1, 2, 3})
	assert.Equal(t, "[1 2 3]", d.String())
}
