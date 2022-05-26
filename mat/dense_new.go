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
func NewDense[T float.DType](rows, cols int, data []T) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if len(data) != rows*cols {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Elements size must be: %d", rows*cols))
	}
	d := densePool[T]().Get(rows, cols)
	copy(d.data, data)
	return d
}

// NewVecDense returns a new column vector (len(data)×1) initialized with
// a copy of raw data.
func NewVecDense[T float.DType](data []T) *Dense[T] {
	d := densePool[T]().Get(len(data), 1)
	copy(d.data, data)
	return d
}

// NewScalar returns a new 1×1 matrix containing the given value.
func NewScalar[T float.DType](v T) *Dense[T] {
	d := densePool[T]().Get(1, 1)
	d.data[0] = v
	return d
}

// NewEmptyVecDense returns a new vector with dimensions size×1, initialized
// with zeros.
func NewEmptyVecDense[T float.DType](size int) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	return densePool[T]().GetEmpty(size, 1)
}

// NewEmptyDense returns a new rows×cols matrix, initialized with zeros.
func NewEmptyDense[T float.DType](rows, cols int) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	return densePool[T]().GetEmpty(rows, cols)
}

// NewOneHotVecDense returns a new one-hot column vector (size×1).
func NewOneHotVecDense[T float.DType](size int, oneAt int) *Dense[T] {
	if size <= 0 {
		panic("mat: the vector size must be a positive number")
	}
	if oneAt < 0 || oneAt >= size {
		panic(fmt.Sprintf("mat: impossible to set the one at index %d. The size is: %d", oneAt, size))
	}
	vec := densePool[T]().GetEmpty(size, 1)
	vec.data[oneAt] = 1
	return vec
}

// NewInitDense returns a new rows×cols dense matrix initialized with a
// constant value.
func NewInitDense[T float.DType](rows, cols int, v T) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	out := densePool[T]().Get(rows, cols)
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = v
	}
	return out
}

// NewInitFuncDense returns a new rows×cols dense matrix initialized with the
// values returned from the callback function.
func NewInitFuncDense[T float.DType](rows, cols int, fn func(r, c int) T) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	out := densePool[T]().Get(rows, cols)

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

	return out
}

// NewInitVecDense returns a new column vector (size×1) initialized with a
// constant value.
func NewInitVecDense[T float.DType](size int, v T) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := densePool[T]().Get(size, 1)
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = v
	}
	return out
}

// NewIdentityDense returns a square identity matrix (size×size), that is,
// with ones on the diagonal and zeros elsewhere.
func NewIdentityDense[T float.DType](size int) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := densePool[T]().GetEmpty(size, size)
	data := out.data
	ln := len(data)
	for i := 0; i < ln; i += size + 1 {
		data[i] = 1
	}
	return out
}
