// Code from `https://stackoverflow.com/questions/31141202/get-the-indices-of-the-array-after-sorting-in-golang`

package utils

import (
	"sort"
)

type Slice struct {
	sort.Interface
	Indices []int
}

func (s Slice) Swap(i, j int) {
	s.Interface.Swap(i, j)
	s.Indices[i], s.Indices[j] = s.Indices[j], s.Indices[i]
}

func NewSlice(n sort.Interface) *Slice {
	s := &Slice{Interface: n, Indices: make([]int, n.Len())}
	for i := range s.Indices {
		s.Indices[i] = i
	}
	return s
}

func NewIntSlice(n ...int) *Slice {
	return NewSlice(sort.IntSlice(n))
}

func NewFloat64Slice(n ...float64) *Slice {
	return NewSlice(sort.Float64Slice(n))
}

func NewStringSlice(n ...string) *Slice {
	return NewSlice(sort.StringSlice(n))
}
