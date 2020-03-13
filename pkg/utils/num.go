// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func SumInt(v []int) (s int) {
	for _, e := range v {
		s += e
	}
	return
}

func ReverseIntSlice(lst []int) []int {
	r := make([]int, len(lst))
	copy(r, lst)
	for i := 0; i < len(r)/2; i++ {
		j := len(r) - i - 1
		r[i], r[j] = r[j], r[i]
	}
	return r
}

func MakeIndices(size int) []int {
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}
	return indices
}

func MakeIntMatrix(rows, cols int) [][]int {
	matrix := make([][]int, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]int, cols)
	}
	return matrix
}

// Abs returns the absolute value of x.
func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
