// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Part of this code is a modified version of Gonum matrix
// pool handling:
// https://github.com/gonum/gonum/blob/master/mat/pool.go
//
// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file available
// at https://github.com/gonum/gonum/blob/master/LICENSE.

package mat

import (
	"fmt"
	"math"
	"math/bits"
	"sync"
)

// DensePool provides pools for slice lengths from 0 to 64 bits
// (0 to MaxUint64).
type DensePool[T DType] [65]sync.Pool

var (
	densePoolFloat32 = NewDensePool[float32]()
	densePoolFloat64 = NewDensePool[float64]()
)

// NewDensePool creates a new pool for handling matrices of a specific DType.
func NewDensePool[T DType]() *DensePool[T] {
	dp := new(DensePool[T])
	for i := range dp {
		dp[i].New = dp.makeDensePoolNewFunction(i)
	}
	return dp
}

// GetDensePool returns the global (sort-of singleton) pre-instantiated pool
// for a specific DType.
func GetDensePool[T DType]() *DensePool[T] {
	// TODO: review this code once stable go 1.18 is released
	switch any(T(0)).(type) {
	case float32:
		return any(densePoolFloat32).(*DensePool[T])
	case float64:
		return any(densePoolFloat64).(*DensePool[T])
	default:
		panic(fmt.Sprintf("mat: no Dense pool for type %T", T(0)))
	}
}

func (dp *DensePool[T]) makeDensePoolNewFunction(bitsLen int) func() any {
	var length uint
	if bitsLen >= 64 {
		length = math.MaxUint64
	} else {
		length = 1<<bitsLen - 1
	}
	return func() any {
		return &Dense[T]{
			rows:  -1,
			cols:  -1,
			flags: denseIsNew | denseIsFromPool,
			data:  make([]T, length),
		}
	}
}

// Get returns a Dense matrix from the pool, with size rows×cols, and
// a raw data slice with a cap in the range rows*cols < cap <= 2*rows*cols.
//
// Warning: the values may not be all zeros. To ensure that all elements
// are initialized to zero, see GetEmptyDense.
func (dp *DensePool[T]) Get(rows, cols int) *Dense[T] {
	d := dp.get(rows, cols)
	d.flags &= ^denseIsNew
	return d
}

// GetEmpty returns a Dense matrix from the pool, with size rows×cols, and
// a raw data slice with a cap in the range rows*cols < cap <= 2*rows*cols.
//
// All elements are guaranteed to be initialized to zero.
func (dp *DensePool[T]) GetEmpty(rows, cols int) *Dense[T] {
	d := dp.get(rows, cols)
	if d.flags&denseIsNew == 0 {
		for i := range d.data {
			d.data[i] = 0
		}
	}
	d.flags &= ^denseIsNew
	return d
}

func (dp *DensePool[T]) get(rows, cols int) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	length := uint(rows * cols)
	bitsLen := bits.Len(length)
	d := dp[bitsLen].Get().(*Dense[T])
	d.data = d.data[:length]
	d.rows = rows
	d.cols = cols
	return d
}

// Put adds a used Dense matrix to pool.
//
// It MUST not be called with a matrix where references to the underlying data
// slice have been kept.
func (dp *DensePool[T]) Put(d *Dense[T]) {
	if d.flags&denseIsFromPool == 0 {
		panic("mat: only matrices originated from the workspace can return to it")
	}
	bitsLen := bits.Len(uint(cap(d.data)))
	dp[bitsLen].Put(d)
}

// ReleaseMatrix puts the given matrix in the appropriate global pool.
// It currently works with Dense matrices only. For any other matrix
// implementation, it panics.
func ReleaseMatrix[T DType](m Matrix[T]) {
	switch mt := m.(type) {
	case *Dense[T]:
		GetDensePool[T]().Put(mt)
	default:
		panic(fmt.Sprintf("mat: cannot release matrix of type %T", mt))
	}
}

// ReleaseDense puts the given matrix in the appropriate global pool.
func ReleaseDense[T DType](m *Dense[T]) {
	GetDensePool[T]().Put(m)
}
