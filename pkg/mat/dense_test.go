// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestDense_AddScalar(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := a.AddScalar(1.0)

	if !floats.EqualApprox(b.Data(), []float64{1.1, 1.2, 1.3, 1.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_AddScalarInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	a.AddScalarInPlace(1.0)

	if !floats.EqualApprox(a.Data(), []float64{1.1, 1.2, 1.3, 1.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_SubScalar(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := a.SubScalar(2.0)

	if !floats.EqualApprox(b.Data(), []float64{-1.9, -1.8, -1.7, -2.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_SubScalarInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	a.SubScalarInPlace(2.0)

	if !floats.EqualApprox(a.Data(), []float64{-1.9, -1.8, -1.7, -2.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_Add(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	c := a.Add(b)

	if !floats.EqualApprox(c.Data(), []float64{0.5, 0.5, 0.8, 0.7}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_AddInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	a.AddInPlace(b)

	if !floats.EqualApprox(a.Data(), []float64{0.5, 0.5, 0.8, 0.7}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_Sub(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	c := a.Sub(b)

	if !floats.EqualApprox(c.Data(), []float64{-0.3, -0.1, -0.2, -0.7}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_SubInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	a.SubInPlace(b)

	if !floats.EqualApprox(a.Data(), []float64{-0.3, -0.1, -0.2, -0.7}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_ProdScalar(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := a.ProdScalar(2.0)

	if !floats.EqualApprox(b.Data(), []float64{0.2, 0.4, 0.6, 0.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_ProdScalarInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	a.ProdScalarInPlace(2.0)

	if !floats.EqualApprox(a.Data(), []float64{0.2, 0.4, 0.6, 0.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_Prod(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	c := a.Prod(b)

	if !floats.EqualApprox(c.Data(), []float64{0.04, 0.06, 0.15, 0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_ProdInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	a.ProdInPlace(b)

	if !floats.EqualApprox(a.Data(), []float64{0.04, 0.06, 0.15, 0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_ProdMatrixScalarInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.0, 0.0, 0.0, 0.0})
	b := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	a.ProdMatrixScalarInPlace(b, 2.0)

	if !floats.EqualApprox(a.Data(), []float64{0.2, 0.4, 0.6, 0.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_Div(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	c := a.Div(b)

	if !floats.EqualApprox(c.Data(), []float64{0.25, 0.6666666666, 0.6, 0.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_DivInPlace(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := NewVecDense([]float64{0.4, 0.3, 0.5, 0.7})
	a.DivInPlace(b)

	if !floats.EqualApprox(a.Data(), []float64{0.25, 0.6666666666, 0.6, 0.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_MulMatrixMatrix(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})
	b := NewDense(4, 3, []float64{
		0.2, 0.7, 0.5,
		0.0, 0.4, 0.5,
		-0.8, 0.7, -0.3,
		0.2, -0.0, -0.9,
	})
	c := a.Mul(b)

	if !floats.EqualApprox(c.Data(), []float64{
		-0.22, 0.36, 0.06,
		0.7, 0.06, 0.0,
		0.52, -0.59, 0.48,
	}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_MulMatrixVector(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})
	b := NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0})
	c := a.Mul(b)

	if !floats.EqualApprox(c.Data(), []float64{-0.53, 0.47, 0.3}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestDense_Pow(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})
	b := a.Pow(3.0)

	if !floats.EqualApprox(b.Data(), []float64{0.001, 0.008, 0.027, 0.0}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestNewDenseNXM(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(a.Data(), []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestNewDenseNXN(t *testing.T) {
	a := NewDense(4, 4, []float64{
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

	if !floats.EqualApprox(a.Data(), []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
		0.9, 0.6, -0.2, 0.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestNewVecDense(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})

	if a.Rows() != 4 || a.Columns() != 1 {
		t.Error("The rows and columns are not correct")
	}

	if !a.IsVector() {
		t.Error("The matrix should be a vector")
	}

	if a.IsScalar() {
		t.Error("The matrix shouldn't be a scalar")
	}

	if !floats.EqualApprox(a.Data(), []float64{0.1, 0.2, 0.3, 0.0}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
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

	if !floats.EqualApprox(a.Data(), []float64{0.42}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
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

	if !floats.EqualApprox(a.Data(), []float64{0.0, 0.0, 0.0, 0.0}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestNewEmptyDenseNXM(t *testing.T) {
	a := NewEmptyDense(3, 4)

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(a.Data(), []float64{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_ZerosLike(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.ZerosLike()

	if b.Rows() != 3 || b.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(b.Data(), []float64{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_OnesLike(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.OnesLike()

	if b.Rows() != 3 || b.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(b.Data(), []float64{
		1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_Zeros(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	a.Zeros()

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(a.Data(), []float64{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestOneHotVecDense(t *testing.T) {
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

	if !floats.EqualApprox(a.Data(), []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestNewInitDenseNXM(t *testing.T) {
	a := NewInitDense(3, 4, 0.42)

	if a.Rows() != 3 || a.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(a.Data(), []float64{
		0.42, 0.42, 0.42, 0.42,
		0.42, 0.42, 0.42, 0.42,
		0.42, 0.42, 0.42, 0.42,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
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

	if !floats.EqualApprox(a.Data(), []float64{0.42, 0.42, 0.42, 0.42}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_Reshape(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.Reshape(4, 3)

	if b.Rows() != 4 || b.Columns() != 3 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(b.Data(), []float64{
		0.1, 0.2, 0.3,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
		0.8, -0.8, -0.1,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_T(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.T()

	if b.Rows() != 4 || b.Columns() != 3 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(b.Data(), []float64{
		0.1, 0.4, -0.5,
		0.2, 0.5, 0.8,
		0.3, -0.6, -0.8,
		0.0, 0.7, -0.1,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_Clone(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.Clone()

	if b.Rows() != 3 || b.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(b.Data(), []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_Copy(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := NewEmptyDense(a.Dims())
	b.Copy(a)

	if b.Rows() != 3 || b.Columns() != 4 {
		t.Error("The rows and columns are not correct")
	}

	if !floats.EqualApprox(b.Data(), []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_SizeMatrix(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	if a.Size() != 12 {
		t.Error("The size is not correct")
	}
}

func TestDense_SizeVector(t *testing.T) {
	a := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0})

	if a.Size() != 4 {
		t.Error("The size is not correct")
	}
}

func TestDense_SizeScalar(t *testing.T) {
	a := NewVecDense([]float64{0.42})

	if a.Size() != 1 {
		t.Error("The size is not correct")
	}
}

func TestDense_Dims(t *testing.T) {
	a := NewDense(3, 4, []float64{
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
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	a.Clip(0.1, 0.3)

	if !floats.EqualApprox(a.Data(), []float64{
		0.1, 0.2, 0.3, 0.1,
		0.3, 0.3, 0.1, 0.3,
		0.1, 0.3, 0.1, 0.1,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_Abs(t *testing.T) {
	a := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})

	b := a.Abs()

	if !floats.EqualApprox(b.Data(), []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, 0.6, 0.7,
		0.5, 0.8, 0.8, 0.1,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestDense_MaxMinSum(t *testing.T) {
	a := NewDense(3, 4, []float64{
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

/*
TODO:
    Range(start, end int) Matrix
    SplitV(sizes ...int) []Matrix
    SumInt(v []int) (s int)
    Set(v float64, i int, j ...int)
    At(i int, j ...int) float64
    Apply(fn func(i, j int, v float64) float64, a Matrix)
    ApplyWithAlpha(fn func(i, j int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64)
    Sqrt() Matrix
*/
