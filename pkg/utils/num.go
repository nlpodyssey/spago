// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import "golang.org/x/exp/rand"

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

// ContainsInt returns whether the list contains the x-element, or not.
func ContainsInt(lst []int, x int) bool {
	for _, element := range lst {
		if element == x {
			return true
		}
	}
	return false
}

// GetUniqueRandomInt generates n mutually exclusive integers up to max, using the default random source.
// The callback checks whether a generated number can be accepted, or not.
func GetUniqueRandomInt(n, max int, valid func(r int) bool) []int {
	a := make([]int, n)
	for i := 0; i < n; i++ {
		r := rand.Intn(max)
		for !valid(r) || ContainsInt(a, r) {
			r = rand.Intn(max)
		}
		a[i] = r
	}
	return a
}

// Abs returns the absolute value of x.
func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
