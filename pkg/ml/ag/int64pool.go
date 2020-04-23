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

package ag

import (
	"sync"
)

// Each pool element i returns slices capped at 1<<i.
// 63 (and not 64) because MaxInt64  = 1<<63 - 1
var int64Pool [63]sync.Pool

func init() {
	for i := range int64Pool {
		length := 1 << uint(i)
		int64Pool[i].New = func() interface{} {
			// Return a pointer type, since it can be put into
			// the return interface value without an allocation.
			return make([]int64, length)
		}
	}
}

// getInt64Slice returns a []int64 with a cap that is less than 2*size.
// Warning, the values may not be at zero.
func getInt64Slice(size int) []int64 {
	return int64Pool[bits(uint64(size))].Get().([]int64)[:size]
}

// releaseInt64Slice replaces a used []int64 into the appropriate size workspace pool.
// Don't call this method if there are any other active references to the slide or its underlying elements.
func releaseInt64Slice(data []int64) {
	int64Pool[bits(uint64(cap(data)))].Put(data)
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
