// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"gonum.org/v1/gonum/floats"
	"math"
	"reflect"
	"testing"
)

func TestSparse_NewSparse(t *testing.T) {
	elements := newTestData()
	s := NewSparse(7, 6, elements)
	if !reflect.DeepEqual(s.nnzRow, []int{0, 2, 4, 7, 8, 8, 9, 10}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.nzElements, []float64{10.0, 20.0, 30.0, 4.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.colsIndex, []int{0, 1, 1, 3, 2, 3, 4, 5, 2, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_NewVecSparse(t *testing.T) {
	elements := newTestDataVec()
	s := NewVecSparse(elements)
	if !reflect.DeepEqual(s.nnzRow, []int{0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.nzElements, []float64{10.0, 3.0, 4.0}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.colsIndex, []int{0, 0, 0}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_NewEmptySparse(t *testing.T) {
	s := NewEmptySparse(7, 6)
	if !reflect.DeepEqual(s.nnzRow, []int{0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.nzElements, []float64{}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.colsIndex, []int{}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_NewEmptyVecSparse(t *testing.T) {
	s := NewEmptyVecSparse(12)
	if !reflect.DeepEqual(s.nnzRow, []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.nzElements, []float64{}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.colsIndex, []int{}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Sparsity(t *testing.T) {
	elements := newTestData()
	s := NewSparse(7, 6, elements)
	sparsity := s.Sparsity()
	if !floats.EqualWithinAbs(sparsity, 0.76190, 0.00001) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_ToDense(t *testing.T) {
	elements := newTestData()
	s := NewSparse(7, 6, elements)
	d := s.ToDense()
	if !floats.EqualApprox(d.Data(), []float64{
		10.0, 20.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 30.0, 0.0, 4.0, 0.0, 0.0,
		0.0, 0.0, 50.0, 60.0, 70.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 80.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 90.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestSparse_Data(t *testing.T) {
	elements := newTestData()
	s := NewSparse(7, 6, elements)
	d := s.Data()
	if !floats.EqualApprox(d, []float64{
		10.0, 20.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 30.0, 0.0, 4.0, 0.0, 0.0,
		0.0, 0.0, 50.0, 60.0, 70.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 80.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 90.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestSparse_Clone(t *testing.T) {
	elements := newTestData()
	s := NewSparse(7, 6, elements)
	d := s.Clone().(*Sparse)
	if !reflect.DeepEqual(d.nnzRow, []int{0, 2, 4, 7, 8, 8, 9, 10}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(d.nzElements, []float64{10.0, 20.0, 30.0, 4.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(d.colsIndex, []int{0, 1, 1, 3, 2, 3, 4, 5, 2, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Copy(t *testing.T) {

	elements := newTestData()
	elements2 := newTestData2()
	s := NewSparse(7, 6, elements)
	d := NewSparse(7, 6, elements2)
	s.Copy(d)

	if !reflect.DeepEqual(s.nnzRow, []int{0, 2, 4, 7, 8, 9, 10, 11}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(s.nzElements, []float64{20.0, 8.0, 30.0, 4.0, 50.0, 60.0, 70.0, 80.0, 25.0, 90.0, 100.0}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(s.colsIndex, []int{1, 5, 1, 3, 2, 3, 4, 5, 2, 2, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_OneHotSparse(t *testing.T) {

	s := OneHotSparse(10, 8)

	if !floats.EqualApprox(s.Data(), []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
	}, 1.0e-6) {
		t.Error("The data doesn't match the expected values")
	}
}

func TestSparse_NewZeros(t *testing.T) {

	s := NewEmptySparse(7, 6)
	s.Zeros()

	if !reflect.DeepEqual(s.nnzRow, []int{0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(s.nzElements, []float64{}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(s.colsIndex, []int{}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_NewAt(t *testing.T) {

	elements := newTestData()
	s := NewSparse(7, 6, elements)
	e := s.At(0, 0)
	f := s.At(6, 4)
	g := s.At(5, 2)

	if e != 10.0 {
		t.Error("The result doesn't match the expected values")
	}

	if f != 0.0 {
		t.Error("The result doesn't match the expected values")
	}

	if g != 90.0 {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_NewAtVec(t *testing.T) {

	elements := newTestDataVec()
	colvector := NewVecSparse(elements)
	rowvector := NewSparse(1, 12, elements)
	e := colvector.AtVec(0)
	f := colvector.AtVec(7)
	g := colvector.AtVec(5)
	h := rowvector.AtVec(0)
	i := rowvector.AtVec(7)
	l := rowvector.AtVec(5)

	if e != 10.0 {
		t.Error("The result doesn't match the expected values")
	}

	if f != 3.0 {
		t.Error("The result doesn't match the expected values")
	}

	if g != 0.0 {
		t.Error("The result doesn't match the expected values")
	}

	if h != 10.0 {
		t.Error("The result doesn't match the expected values")
	}

	if i != 3.0 {
		t.Error("The result doesn't match the expected values")
	}

	if l != 0.0 {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_ProdScalar(t *testing.T) {

	elements := newTestData()
	s := NewSparse(7, 6, elements)

	d := s.ProdScalar(3.0).(*Sparse)

	if !reflect.DeepEqual(d.nnzRow, []int{0, 2, 4, 7, 8, 8, 9, 10}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.nzElements, []float64{30.0, 60.0, 90.0, 12.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.colsIndex, []int{0, 1, 1, 3, 2, 3, 4, 5, 2, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_ProdScalarInPlace(t *testing.T) {

	elements := newTestData()
	s := NewSparse(7, 6, elements)

	d := s.ProdScalarInPlace(3.0).(*Sparse)

	if !reflect.DeepEqual(d.nnzRow, []int{0, 2, 4, 7, 8, 8, 9, 10}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.nzElements, []float64{30.0, 60.0, 90.0, 12.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.colsIndex, []int{0, 1, 1, 3, 2, 3, 4, 5, 2, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_ProdMatrixScalarInPlace(t *testing.T) {

	elements := newTestData()
	s := NewSparse(7, 6, elements)
	d := NewEmptySparse(7, 6)
	d = d.ProdMatrixScalarInPlace(s, 3.0).(*Sparse)

	if !reflect.DeepEqual(d.nnzRow, []int{0, 2, 4, 7, 8, 8, 9, 10}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.nzElements, []float64{30.0, 60.0, 90.0, 12.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.colsIndex, []int{0, 1, 1, 3, 2, 3, 4, 5, 2, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_AddScalar(t *testing.T) {

	s := NewSparse(3, 4, newTestDataD())
	r := s.AddScalar(0.5)

	if !floats.EqualApprox(r.Data(), []float64{
		0.5, 0.7, 0.5, 0.5,
		0.5, 0.8, 0.5, 0.3,
		0.5, 0.5, 0.0, 0.5}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_SubScalar(t *testing.T) {

	s := NewSparse(3, 4, newTestDataD())
	r := s.SubScalar(0.5)

	if !floats.EqualApprox(r.Data(), []float64{
		-0.5, -0.3, -0.5, -0.5,
		-0.5, -0.2, -0.5, -0.7,
		-0.5, -0.5, -1.0, -0.5}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Add(t *testing.T) {

	// sparse dense
	d := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})
	s := NewSparse(3, 4, newTestDataD())
	r := s.Add(d)

	if !floats.EqualApprox(r.Data(), []float64{
		0.1, 0.4, 0.3, 0.0,
		0.4, 0.8, -0.6, 0.5,
		-0.5, 0.8, -1.3, -0.1}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	// sparse sparse
	s1 := NewSparse(3, 4, newTestDataD())
	s2 := NewSparse(3, 4, newTestDataE())

	u := s1.Add(s2).(*Sparse)
	if !reflect.DeepEqual(u.nnzRow, []int{0, 2, 3, 6}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(u.nzElements, []float64{0.2, 0.3, -0.4, 2.0, -0.5, 1.0}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(u.colsIndex, []int{1, 3, 3, 0, 2, 3}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Sub(t *testing.T) {

	// sparse dense
	d := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})
	s := NewSparse(3, 4, newTestDataD())
	r := s.Sub(d)

	if !floats.EqualApprox(r.Data(), []float64{
		-0.1, 0.0, -0.3, 0.0,
		-0.4, -0.2, 0.6, -0.9,
		0.5, -0.8, 0.3, 0.1}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	// sparse sparse
	s1 := NewSparse(3, 4, newTestDataD())
	s2 := NewSparse(3, 4, newTestDataE())

	u := s1.Sub(s2).(*Sparse)
	if !reflect.DeepEqual(u.nnzRow, []int{0, 2, 3, 6}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(u.nzElements, []float64{0.2, -0.3, 0.6, -2.0, -0.5, -1.0}) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(u.colsIndex, []int{1, 3, 1, 0, 2, 3}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Prod(t *testing.T) {

	// sparse dense
	d := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})
	s := NewSparse(3, 4, newTestDataD())
	r := s.Prod(d).(*Sparse)

	if !reflect.DeepEqual(r.nnzRow, []int{0, 1, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(r.nzElements, []float64{0.04, 0.15, -0.14, 0.4}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(r.colsIndex, []int{1, 1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}

	// sparse sparse
	s1 := NewSparse(3, 4, newTestDataD())
	s2 := NewSparse(3, 4, newTestDataE())

	u := s1.Prod(s2).(*Sparse)
	if !reflect.DeepEqual(u.nnzRow, []int{0, 0, 2, 2}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(u.nzElements, []float64{-0.09, 0.04}, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(u.colsIndex, []int{1, 3}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Div(t *testing.T) {

	// sparse dense
	d := NewDense(3, 4, []float64{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	})
	s := NewSparse(3, 4, newTestDataD())
	r := s.Div(d).(*Sparse)

	if !reflect.DeepEqual(r.nnzRow, []int{0, 1, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(r.nzElements, []float64{1.0, 0.6, -0.285714, 0.625}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(r.colsIndex, []int{1, 1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Mul(t *testing.T) {

	// sparse dense
	b := NewDense(4, 3, []float64{
		0.2, 0.7, 0.5,
		0.0, 0.4, 0.5,
		-0.8, 0.7, -0.3,
		0.2, -0.0, -0.9,
	})
	s := NewSparse(3, 4, newTestDataD())
	r := s.Mul(b)

	if !floats.EqualApprox(r.Data(), []float64{
		0.0, 0.08, 0.1,
		-0.04, 0.12, 0.33,
		0.4, -0.35, 0.15,
	}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	//sparse sparse
	s1 := NewSparse(3, 4, newTestDataD())
	s2 := NewSparse(4, 3, newTestDataF())
	u := s1.Mul(s2)

	if !floats.EqualApprox(u.Data(), []float64{
		0.04, 0.0, 0.0,
		0.08, 0.0, -0.04,
		0.0, 0.0, -0.45,
	}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_DotUnitary(t *testing.T) {
	// sparse dense

	c := NewVecDense([]float64{0.1, 0.2, 0.3, 0.0, 0.4, 0.8})
	d := NewSparse(1, 6, []float64{0.0, 0.0, 0.0, 0.7, 0.1, 0.0})
	u := d.DotUnitary(c)

	if !floats.EqualWithinAbs(u, 0.04, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}

	// sparse Sparse

	e := NewSparse(1, 6, []float64{0.0, 0.0, 0.3, 0.0, 0.9, 0.0})
	f := NewSparse(1, 6, []float64{0.0, 0.0, 0.0, 0.7, 0.1, 0.0})
	v := e.DotUnitary(f)

	if !floats.EqualWithinAbs(v, 0.09, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Transpose(t *testing.T) {

	s := NewSparse(3, 4, newTestDataD())
	r := s.T().(*Sparse)

	if !reflect.DeepEqual(r.nnzRow, []int{0, 0, 2, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(r.nzElements, []float64{0.2, 0.3, -0.5, -0.2}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(r.colsIndex, []int{0, 1, 2, 1}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Pow(t *testing.T) {

	elements := newTestDataD()
	s := NewSparse(3, 4, elements)

	d := s.Pow(3.0).(*Sparse)

	if !reflect.DeepEqual(d.nnzRow, []int{0, 1, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(d.nzElements, []float64{0.008, 0.027, -0.008, -0.125}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.colsIndex, []int{1, 1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Sqrt(t *testing.T) {

	elements := newTestDataG()
	s := NewSparse(3, 4, elements)

	d := s.Sqrt().(*Sparse)

	if !reflect.DeepEqual(d.nnzRow, []int{0, 1, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(d.nzElements, []float64{0.447213, 0.547722, 0.447213, 0.547722}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.colsIndex, []int{1, 1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Abs(t *testing.T) {

	elements := newTestDataD()
	s := NewSparse(3, 4, elements)

	d := s.Abs().(*Sparse)

	if !reflect.DeepEqual(d.nnzRow, []int{0, 1, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(d.nzElements, []float64{0.2, 0.3, 0.2, 0.5}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(d.colsIndex, []int{1, 1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Clip(t *testing.T) {

	elements := newTestDataD()
	s := NewSparse(3, 4, elements)

	s.ClipInPlace(0.1, 0.2)

	if !reflect.DeepEqual(s.nnzRow, []int{0, 1, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}

	if !floats.EqualApprox(s.nzElements, []float64{0.2, 0.2, 0.1, 0.1}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	if !reflect.DeepEqual(s.colsIndex, []int{1, 1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Norm(t *testing.T) {

	elements := newTestDataD()
	s := NewSparse(3, 4, elements)

	d := s.Norm(2)

	if !floats.EqualWithinAbs(d, 0.648074, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Sum(t *testing.T) {
	elements := newTestDataD()
	s := NewSparse(3, 4, elements)
	d := s.Sum()
	if !floats.EqualWithinAbs(d, -0.2, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Max(t *testing.T) {
	elements := newTestDataD()
	s := NewSparse(3, 4, elements)
	d := s.Max()
	if !floats.EqualWithinAbs(d, 0.3, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Min(t *testing.T) {
	elements := newTestDataD()
	s := NewSparse(3, 4, elements)
	d := s.Min()
	if !floats.EqualWithinAbs(d, -0.5, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Apply(t *testing.T) {
	elements := newTestDataD()
	s := NewSparse(3, 4, elements)
	s.Apply(func(i, j int, v float64) float64 {
		return math.Sin(v)
	}, s)
	if !reflect.DeepEqual(s.nnzRow, []int{0, 1, 3, 4}) {
		t.Error("The result doesn't match the expected values")
	}
	if !floats.EqualApprox(s.nzElements, []float64{0.198669, 0.29552, -0.198669, -0.479425}, 1.0e-5) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(s.colsIndex, []int{1, 1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Maximum(t *testing.T) {
	s1 := NewSparse(3, 4, newTestDataD())
	s2 := NewSparse(3, 4, newTestDataE())
	u := s1.Maximum(s2).(*Sparse)
	if !reflect.DeepEqual(u.nnzRow, []int{0, 2, 4, 6}) {
		t.Error("The result doesn't match the expected values")
	}
	if !floats.EqualApprox(u.nzElements, []float64{0.2, 0.3, 0.3, -0.2, 2.0, 1.0}, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(u.colsIndex, []int{1, 3, 1, 3, 0, 3}) {
		t.Error("The result doesn't match the expected values")
	}
}

func TestSparse_Minimum(t *testing.T) {
	s1 := NewSparse(3, 4, newTestDataD())
	s2 := NewSparse(3, 4, newTestDataE())
	u := s1.Minimum(s2).(*Sparse)
	if !reflect.DeepEqual(u.nnzRow, []int{0, 0, 2, 3}) {
		t.Error("The result doesn't match the expected values")
	}
	if !floats.EqualApprox(u.nzElements, []float64{-0.3, -0.2, -0.5}, 1e-06) {
		t.Error("The result doesn't match the expected values")
	}
	if !reflect.DeepEqual(u.colsIndex, []int{1, 3, 2}) {
		t.Error("The result doesn't match the expected values")
	}
}

///////////////////
//  Testing data
///////////////////
func newTestData() []float64 {
	out := []float64{
		10.0, 20.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 30.0, 0.0, 4.0, 0.0, 0.0,
		0.0, 0.0, 50.0, 60.0, 70.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 80.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 90.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
	}
	return out
}

// Testing data
func newTestData2() []float64 {
	out := []float64{
		0.0, 20.0, 0.0, 0.0, 0.0, 8.0,
		0.0, 30.0, 0.0, 4.0, 0.0, 0.0,
		0.0, 0.0, 50.0, 60.0, 70.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 80.0,
		0.0, 0.0, 25.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 90.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
	}
	return out
}

// Testing data
func newTestDataD() []float64 {
	out := []float64{
		0.0, 0.2, 0.0, 0.0,
		0.0, 0.3, 0.0, -0.2,
		0.0, 0.0, -0.5, 0.0,
	}
	return out
}

func newTestDataE() []float64 {
	out := []float64{
		0.0, 0.0, 0.0, 0.3,
		0.0, -0.3, 0.0, -0.2,
		2.0, 0.0, 0.0, 1.0,
	}
	return out
}

func newTestDataF() []float64 {
	out := []float64{
		0.0, 0.3, 0.0,
		0.2, 0.0, 0.0,
		0.0, 0.0, 0.9,
		-0.1, 0.0, 0.2,
	}
	return out
}

func newTestDataG() []float64 {
	out := []float64{
		0.0, 0.2, 0.0, 0.0,
		0.0, 0.3, 0.0, 0.2,
		0.0, 0.0, 0.3, 0.0,
	}
	return out
}

func newTestDataVec() []float64 {
	out := []float64{
		10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0,
	}
	return out
}
