// Code from `https://stackoverflow.com/questions/31141202/get-the-indices-of-the-array-after-sorting-in-golang`

package utils

import (
	"sort"
)

// Slice is an extension of sort.Interface which keeps track of the
// original indices of the elements of a slice after sorting.
type Slice struct {
	sort.Interface
	Indices []int
}

// Swap swaps the elements with indexes i and j.
func (s Slice) Swap(i, j int) {
	s.Interface.Swap(i, j)
	s.Indices[i], s.Indices[j] = s.Indices[j], s.Indices[i]
}

// NewSlice returns a new Slice.
func NewSlice(n sort.Interface) *Slice {
	s := &Slice{Interface: n, Indices: make([]int, n.Len())}
	for i := range s.Indices {
		s.Indices[i] = i
	}
	return s
}

// NewIntSlice returns a new Slice for the given sequence of int values.
func NewIntSlice(n ...int) *Slice {
	return NewSlice(sort.IntSlice(n))
}

// NewFloat64Slice returns a new Slice for the given sequence of float64 values.
func NewFloat64Slice(n ...float64) *Slice {
	return NewSlice(sort.Float64Slice(n))
}

// NewStringSlice returns a new Slice for the given sequence of string values.
func NewStringSlice(n ...string) *Slice {
	return NewSlice(sort.StringSlice(n))
}
