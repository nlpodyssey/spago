// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

import stdsort "sort"

// FloatSlice implements the standard library's sort.Interface for any slice of
// floating-point numbers. The values are sorted in increasing order, and
// nonnumerical NaN (Not a Number) values are ordered before other numerical
// values.
type FloatSlice[F Float] []F

// Len is the number of elements in the collection.
func (x FloatSlice[_]) Len() int {
	return len(x)
}

// Less reports whether the value at index i must be ordered before the
// value at index j, as required by the the standard library's sort.Interface.
//
// Floating-point comparisons are not transitive relations: there is no
// consistent ordering for NaN (Not a Number) values.
// This implementation places nonnumerical NaN values before other numerical
// values.
func (x FloatSlice[_]) Less(i, j int) bool {
	return x[i] < x[j] || (isNaN(x[i]) && !isNaN(x[j]))
}

// Swap swaps the elements with indexes i and j.
func (x FloatSlice[_]) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

// Sort is a convenience method: x.Sort() calls sort.Sort(x).
func (x FloatSlice[_]) Sort() {
	stdsort.Sort(x)
}

// ReverseSort is a convenience method: x.Sort() calls sort.Sort(sort.Reverse(x)).
func (x FloatSlice[_]) ReverseSort() {
	stdsort.Sort(stdsort.Reverse(x))
}

// Sort sorts a slice of floating-point numbers in ascending order.
func Sort[F Float](x []F) {
	FloatSlice[F](x).Sort()
}

// ReverseSort sorts a slice of floating-point numbers in descending order.
func ReverseSort[F Float](x []F) {
	FloatSlice[F](x).ReverseSort()
}

// isNaN is a copy of math.IsNaN to avoid a dependency on the math package.
func isNaN[F Float](f F) bool {
	return f != f
}
