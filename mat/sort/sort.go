// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

import (
	"github.com/nlpodyssey/spago/mat"
	"sort"
)

// DTSlice attaches the methods of sort.Interface to []T, sorting in increasing order
// (not-a-number values are treated as less than other values).
type DTSlice[T mat.DType] []T

func (p DTSlice[_]) Len() int           { return len(p) }
func (p DTSlice[_]) Less(i, j int) bool { return p[i] < p[j] || isNaN(p[i]) && !isNaN(p[j]) }
func (p DTSlice[_]) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// isNaN is a copy of math.IsNaN.
func isNaN[T mat.DType](f T) bool {
	return f != f
}

// Sort is a convenience method.
func (p DTSlice[_]) Sort() { sort.Sort(p) }
