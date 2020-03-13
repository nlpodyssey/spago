// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
	"io"
	"math"
)

var _ Matrix = &Sparse{}

type Sparse struct {
	delegate *sparse.DOK
}

func NewVecSparse(size int) *Sparse {
	return &Sparse{sparse.NewDOK(size, 1)}
}

func NewSparse(r, c int) *Sparse {
	return &Sparse{sparse.NewDOK(r, c)}
}

func OneHotSparse(size int, oneAt int) *Sparse {
	vec := NewVecSparse(size)
	vec.Set(1.0, oneAt)
	return vec
}

func (s *Sparse) SetData(data []float64) {
	// TODO
}

func (s *Sparse) ZerosLike() Matrix {
	return NewSparse(s.Dims())
}

func (s *Sparse) OnesLike() Matrix {
	buf := s.ZerosLike().(*Sparse)
	s.delegate.DoNonZero(func(i, j int, v float64) {
		buf.delegate.Set(i, j, 1.0)
	})
	return buf
}

func (s *Sparse) Clone() Matrix {
	buf := NewSparse(s.delegate.Dims())
	s.delegate.DoNonZero(func(i, j int, v float64) {
		buf.delegate.Set(i, j, v)
	})
	return buf
}

func (s *Sparse) Copy(other Matrix) {
	s.Zeros()
	switch other := other.(type) {
	case *Sparse:
		other.delegate.DoNonZero(func(i, j int, v float64) {
			s.delegate.Set(i, j, v)
		})
	default:
		panic("mat: unsupported matrix")
	}
}

func (s *Sparse) Zeros() {
	s.delegate.DoNonZero(func(i, j int, v float64) {
		s.delegate.Set(i, j, 0)
	})
}

func (s *Sparse) Dims() (r, c int) {
	return s.delegate.Dims()
}

func (s *Sparse) Rows() int {
	r, _ := s.delegate.Dims()
	return r
}

func (s *Sparse) Columns() int {
	_, c := s.delegate.Dims()
	return c
}

func (s *Sparse) Size() int {
	r, c := s.delegate.Dims()
	return r * c
}

func (s *Sparse) LastIndex() int {
	return s.Size() - 1
}

func (s *Sparse) Data() []float64 {
	return s.delegate.ToDense().RawMatrix().Data
}

func (s *Sparse) IsVector() bool {
	return s.Rows() == 1 || s.Columns() == 1
}

func (s *Sparse) IsScalar() bool {
	return s.Size() == 1
}

func (s *Sparse) Scalar() float64 {
	if !s.IsScalar() {
		panic("mat: expected scalar but the matrix contains more elements.")
	}
	return s.At(0)
}

func (s *Sparse) Set(v float64, i int, j ...int) {
	if len(j) > 1 {
		panic("mat: invalid 'j' argument.")
	}
	if len(j) > 0 {
		s.delegate.Set(i, j[0], v)
	} else {
		s.delegate.Set(i, 0, v)
	}
}

func (s *Sparse) At(i int, j ...int) float64 {
	if len(j) > 1 {
		panic("mat: invalid 'j' argument.")
	}
	if len(j) > 0 {
		return s.delegate.At(i, j[0])
	} else {
		return s.delegate.At(i, 0)
	}
}

func (s *Sparse) DoNonZero(fn func(i, j int, v float64)) {
	s.delegate.DoNonZero(func(i, j int, v float64) {
		fn(i, j, v)
	})
}

func (s *Sparse) T() Matrix {
	r, c := s.Dims()
	m := NewSparse(c, r)
	s.DoNonZero(func(i, j int, v float64) {
		m.Set(s.At(i, j), j, i)
	})
	return m
}

func (s *Sparse) Reshape(r, c int) Matrix {
	panic("mat: Reshape not implemented for sparse matrix")
}

func (s *Sparse) Apply(fn func(i, j int, v float64) float64, a Matrix) {
	panic("mat: Apply not implemented for sparse matrix")
}

func (s *Sparse) ApplyWithAlpha(fn func(i, j int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64) {
	panic("mat: Apply not implemented for sparse matrix")
}

func (s *Sparse) AddScalar(n float64) Matrix {
	panic("mat: AddScalar not implemented for sparse matrix")
}

func (s *Sparse) SubScalar(n float64) Matrix {
	panic("mat: SubScalar not implemented for sparse matrix")
}

func (s *Sparse) AddScalarInPlace(n float64) Matrix {
	panic("mat: AddScalarInPlace not implemented for sparse matrix")
}

func (s *Sparse) SubScalarInPlace(n float64) Matrix {
	panic("mat: SubScalarInPlace not implemented for sparse matrix")
}

func (s *Sparse) ProdScalarInPlace(n float64) Matrix {
	s.delegate.DoNonZero(func(i, j int, v float64) {
		s.delegate.Set(i, j, v*n)
	})
	return s
}

func (s *Sparse) ProdMatrixScalarInPlace(m Matrix, n float64) Matrix {
	panic("mat: ProdMatrixScalarInPlace not implemented for sparse matrix")
}

func (s *Sparse) ProdScalar(n float64) Matrix {
	out := s.Clone().(*Sparse)
	out.ProdScalarInPlace(n)
	return out
}

func (s *Sparse) Add(other Matrix) Matrix {
	panic("mat: Add not implemented for sparse matrix")
}

func (s *Sparse) AddInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		other.delegate.DoNonZero(func(i, j int, v float64) {
			s.delegate.Set(i, j, s.delegate.At(i, j)+v)
		})
	default:
		panic("mat: unsupported matrix")
	}
	return s
}

func (s *Sparse) Sub(other Matrix) Matrix {
	panic("mat: Sub not implemented for sparse matrix")
}

func (s *Sparse) SubInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		other.delegate.DoNonZero(func(i, j int, v float64) {
			s.delegate.Set(i, j, s.delegate.At(i, j)-v)
		})
	default:
		panic("mat: unsupported matrix")
	}
	return s
}

func (s *Sparse) Prod(other Matrix) Matrix {
	panic("mat: Prod not implemented for sparse matrix")
}

func (s *Sparse) ProdInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		other.delegate.DoNonZero(func(i, j int, v float64) {
			s.delegate.Set(i, j, s.delegate.At(i, j)*v)
		})
	default:
		panic("mat: unsupported matrix")
	}
	return s
}

func (s *Sparse) Div(other Matrix) Matrix {
	panic("mat: Div not implemented for sparse matrix")
}

func (s *Sparse) DivInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		other.delegate.DoNonZero(func(i, j int, v float64) {
			s.delegate.Set(i, j, s.delegate.At(i, j)/v)
		})
	default:
		panic("mat: unsupported matrix")
	}
	return s
}

func (s *Sparse) Mul(other Matrix) Matrix {
	panic("mat: Mul not implemented for sparse matrix")
}

func (s *Sparse) DotUnitary(other Matrix) float64 {
	panic("mat: DotUnitary not implemented for sparse matrix")
}

func (s *Sparse) Pow(power float64) Matrix {
	r, c := s.Dims()
	m := NewSparse(c, r)
	s.DoNonZero(func(i, j int, v float64) {
		m.Set(math.Pow(s.At(i, j), power), i, j)
	})
	return m
}

func (s *Sparse) Sqrt() Matrix {
	return nil // TODO
}

func (s *Sparse) Norm(pow float64) float64 {
	return 0.0 // TODO
}

func (s *Sparse) ClipInPlace(min, max float64) Matrix {
	s.DoNonZero(func(i, j int, v float64) {
		if s.At(i, j) < min {
			s.Set(min, i, j)
		} else if s.At(i, j) > max {
			s.Set(max, i, j)
		}
	})
	return s
}

func (s *Sparse) Abs() Matrix {
	r, c := s.Dims()
	m := NewSparse(c, r)
	s.DoNonZero(func(i, j int, v float64) {
		m.Set(math.Abs(s.At(i, j)), i, j)
	})
	return m
}

func (s *Sparse) Sum() float64 {
	return mat.Sum(s.delegate)
}

func (s *Sparse) Max() float64 {
	return mat.Max(s.delegate)
}

func (s *Sparse) Min() float64 {
	return mat.Min(s.delegate)
}

func (s *Sparse) MarshalBinaryTo(w io.Writer) (int, error) {
	return s.delegate.MarshalBinaryTo(w)
}

func (s *Sparse) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	return s.delegate.UnmarshalBinaryFrom(r)
}

func (s *Sparse) String() string {
	return fmt.Sprintf("%v", s.delegate)
}
