// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

// MinInt returns the minimum value between a and b.
func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SumInt returns the sum of all values from v.
func SumInt(v []int) (s int) {
	for _, e := range v {
		s += e
	}
	return
}

// ReverseIntSlice returns a reversed version of the given slice.
func ReverseIntSlice(lst []int) []int {
	r := make([]int, len(lst))
	copy(r, lst)
	for i := 0; i < len(r)/2; i++ {
		j := len(r) - i - 1
		r[i], r[j] = r[j], r[i]
	}
	return r
}

// MakeIndices returns a slice of the given size, where each element has
// the same value of its own index position.
func MakeIndices(size int) []int {
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}
	return indices
}

// MakeIntMatrix returns a new 2-dimensional slice of int.
func MakeIntMatrix(rows, cols int) [][]int {
	matrix := make([][]int, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]int, cols)
	}
	return matrix
}

// ContainsInt returns whether the list contains the x-element, or not.
func ContainsInt(lst []int, x int) bool {
	for _, element := range lst {
		if element == x {
			return true
		}
	}
	return false
}

// IntSliceEqual returns whether the two slices are equal, or not.
func IntSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, va := range a {
		if va != b[i] {
			return false
		}
	}
	return true
}

// GetNeighborsIndices returns the indices of the (circular) neighbours at position index.
func GetNeighborsIndices(size, index, windowSize int) []int {
	low := index - windowSize
	high := index + windowSize
	indices := make([]int, 2*windowSize)
	for i := 0; i < len(indices); i++ {
		if low < 0 {
			indices[i] = size + low
			low++
		} else if high > size {
			indices[i] = high - size - 1
			high--
		} else {
			indices[i] = low
			low++
		}
	}
	return indices
}

// Abs returns the absolute value of x.
func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
