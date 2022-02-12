// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

type intSet map[int]struct{}

func newIntSet() intSet {
	return make(intSet)
}

// Add adds a value to the set.
func (s intSet) Add(i int) {
	s[i] = struct{}{}
}

// Delete deletes a value from the set, if it exists.
func (s intSet) Delete(i int) {
	delete(s, i)
}

// Has reports whether the set contains the given value.
func (s intSet) Has(i int) bool {
	_, ok := s[i]
	return ok
}
