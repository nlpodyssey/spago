// Copyright 2020 spaGO Authors. All rights reserved.
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

package mat32

import (
	"sync"
)

// TODO: adapt Dense Workspace to 32bits Float

// Each pool element i returns slices capped at 1<<i.
// 63 (and not 64) because MaxInt64  = 1<<63 - 1
var densePool [63]sync.Pool

//var densePool [63]*utils.Pool // alternative to sync.Pool

func init() {
	for i := range densePool {
		length := 1 << uint(i)
		//densePool[i] = utils.NewPool(10000) // enable if you're using utils.Pool
		densePool[i].New = func() interface{} {
			// Return a pointer type, since it can be put into
			// the return interface value without an allocation.
			return &Dense{
				rows:     -1,
				cols:     -1,
				size:     -1,
				data:     make([]Float, length),
				viewOf:   nil,
				fromPool: true,
			}
		}
	}
}

// GetDenseWorkspace returns a *Dense of size r×c and a data slice with a cap that is less than 2*r*c.
// Warning, the values may not be at zero. If you need a ready-to-use matrix you can call GetEmptyDenseWorkspace().
func GetDenseWorkspace(r, c int) *Dense {
	size := r * c
	w := densePool[bits(uint64(size))].Get().(*Dense)
	w.data = w.data[:size]
	w.rows = r
	w.cols = c
	w.size = size
	return w
}

// GetEmptyDenseWorkspace returns a *Dense of size r×c and a data slice with a cap that is less than 2*r*c.
// The returned matrix is ready-to-use (with all the values set to zeros).
func GetEmptyDenseWorkspace(r, c int) *Dense {
	size := r * c
	i := bits(uint64(size))
	w := densePool[i].Get().(*Dense)
	isNew := w.size == -1 // only a new matrix has size -1
	w.data = w.data[:size]
	w.rows = r
	w.cols = c
	w.size = size
	if !isNew {
		zero(w.data)
	}
	return w
}

// ReleaseMatrix checks whether m is a Dense matrix, and, if so, it
// releases is, otherwise no operation is performed.
func ReleaseMatrix(m Matrix) {
	d, isDense := m.(*Dense)
	if !isDense {
		return
	}
	ReleaseDense(d)
}

// ReleaseDense replaces a used *Dense into the appropriate size
// workspace pool. ReleaseDense must not be called with a matrix
// where references to the underlying data slice have been kept.
func ReleaseDense(w *Dense) {
	if !w.fromPool {
		panic("mat32: only matrices originated from the workspace can return to it")
	}
	densePool[bits(uint64(cap(w.data)))].Put(w)
}

var tab64 = [64]byte{
	0x3f, 0x00, 0x3a, 0x01, 0x3b, 0x2f, 0x35, 0x02,
	0x3c, 0x27, 0x30, 0x1b, 0x36, 0x21, 0x2a, 0x03,
	0x3d, 0x33, 0x25, 0x28, 0x31, 0x12, 0x1c, 0x14,
	0x37, 0x1e, 0x22, 0x0b, 0x2b, 0x0e, 0x16, 0x04,
	0x3e, 0x39, 0x2e, 0x34, 0x26, 0x1a, 0x20, 0x29,
	0x32, 0x24, 0x11, 0x13, 0x1d, 0x0a, 0x0d, 0x15,
	0x38, 0x2d, 0x19, 0x1f, 0x23, 0x10, 0x09, 0x0c,
	0x2c, 0x18, 0x0f, 0x08, 0x17, 0x07, 0x06, 0x05,
}

// bits returns the ceiling of base 2 log of v.
// Approach based on http://stackoverflow.com/a/11398748.
func bits(v uint64) byte {
	if v == 0 {
		return 0
	}
	v <<= 2
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	return tab64[((v-(v>>1))*0x07EDD5E59A4E28C2)>>58] - 1
}

// zero zeros the given slice's elements.
func zero(f []Float) {
	for i := range f {
		f[i] = 0.0
	}
}
