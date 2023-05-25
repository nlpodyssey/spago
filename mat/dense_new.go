// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat/float"
)

// NewDense returns a new matrix of size rows×cols, initialized with a
// copy of raw data.
//
// Rows and columns MUST not be negative, and the length of data MUST be
// equal to rows*cols, otherwise the method panics.
func NewDense[T float.DType](rows, cols int, data []T, opts ...MatrixOption) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if len(data) != rows*cols {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Elements size must be: %d", rows*cols))
	}
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	d := makeDense[T](rows, cols, malloc[T](len(data)))
	copy(d.data, data)

	for _, opt := range opts {
		opt(d)
	}
	return d
}

// NewVecDense returns a new column vector (len(data)×1) initialized with
// a copy of raw data.
func NewVecDense[T float.DType](data []T, opts ...MatrixOption) *Dense[T] {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	d := makeDense[T](len(data), 1, malloc[T](len(data)))
	copy(d.data, data)

	for _, opt := range opts {
		opt(d)
	}
	return d
}

// Scalar returns a new 1×1 matrix containing the given value.
func Scalar[T float.DType](v T, opts ...MatrixOption) *Dense[T] {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	d := makeDense[T](1, 1, malloc[T](1))
	d.data[0] = v

	for _, opt := range opts {
		opt(d)
	}
	return d
}

// NewEmptyVecDense returns a new vector with dimensions size×1, initialized
// with zeros.
func NewEmptyVecDense[T float.DType](size int, opts ...MatrixOption) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	d := makeDense[T](size, 1, malloc[T](size))
	for _, opt := range opts {
		opt(d)
	}
	return d
}

// NewEmptyDense returns a new rows×cols matrix, initialized with zeros.
func NewEmptyDense[T float.DType](rows, cols int, opts ...MatrixOption) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	d := makeDense[T](rows, cols, malloc[T](rows*cols))

	for _, opt := range opts {
		opt(d)
	}
	return d
}

// NewOneHotVecDense returns a new one-hot column vector (size×1).
func NewOneHotVecDense[T float.DType](size int, oneAt int, opts ...MatrixOption) *Dense[T] {
	if size <= 0 {
		panic("mat: the vector size must be a positive number")
	}
	if oneAt < 0 || oneAt >= size {
		panic(fmt.Sprintf("mat: impossible to set the one at index %d. The size is: %d", oneAt, size))
	}
	vec := makeDense[T](size, 1, malloc[T](size))
	vec.data[oneAt] = 1

	for _, opt := range opts {
		opt(vec)
	}
	return vec
}

// NewInitDense returns a new rows×cols dense matrix initialized with a
// constant value.
func NewInitDense[T float.DType](rows, cols int, v T, opts ...MatrixOption) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](rows, cols, malloc[T](rows*cols))
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = v
	}
	for _, opt := range opts {
		opt(out)
	}
	return out
}

// NewInitFuncDense returns a new rows×cols dense matrix initialized with the
// values returned from the callback function.
func NewInitFuncDense[T float.DType](rows, cols int, fn func(r, c int) T, opts ...MatrixOption) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](rows, cols, malloc[T](rows*cols))

	outData := out.data

	r := 0
	c := 0
	for i := range outData {
		outData[i] = fn(r, c)
		c++
		if c == cols {
			r++
			c = 0
		}
	}

	for _, opt := range opts {
		opt(out)
	}
	return out
}

// NewInitVecDense returns a new column vector (size×1) initialized with a
// constant value.
func NewInitVecDense[T float.DType](size int, v T, opts ...MatrixOption) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](size, 1, malloc[T](size))
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = v
	}

	for _, opt := range opts {
		opt(out)
	}
	return out
}

// NewIdentityDense returns a square identity matrix (size×size), that is,
// with ones on the diagonal and zeros elsewhere.
func NewIdentityDense[T float.DType](size int, opts ...MatrixOption) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := makeDense[T](size, size, malloc[T](size*size))
	data := out.data
	ln := len(data)
	for i := 0; i < ln; i += size + 1 {
		data[i] = 1
	}

	for _, opt := range opts {
		opt(out)
	}
	return out
}
