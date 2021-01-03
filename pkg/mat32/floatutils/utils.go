// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package floatutils

import (
	"github.com/nlpodyssey/spago/pkg/mat32/internal"
	"math"
	"strconv"
	"strings"
)

// EqualApprox returns true if a and b are equal to within reasonable
// absolute tolerance (hardcoded as 1.0e-04).
func EqualApprox(a, b float32) bool {
	return a == b || float32(math.Abs(float64(a-b))) <= 1.0e-04
}

// SliceEqualApprox returns true if a and b have the same length and EqualApprox
// is true for each element pair from a and b.
func SliceEqualApprox(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, va := range a {
		if !EqualApprox(va, b[i]) {
			return false
		}
	}
	return true
}

// Copy creates and return a copy of the given slice.
func Copy(in []float32) []float32 {
	out := make([]float32, len(in))
	copy(out, in)
	return out
}

// FillFloatSlice fills the given slice's elements with value.
func FillFloatSlice(slice []float32, value float32) {
	for i := range slice {
		slice[i] = value
	}
}

// Sign returns +1 if a is positive, -1 if a is negative, or 0 if a is 0.
func Sign(a float32) int {
	switch {
	case a < 0:
		return -1
	case a > 0:
		return +1
	}
	return 0
}

// Max returns the maximum value from the given slice, which MUST NOT be empty.
func Max(v []float32) (m float32) {
	m = v[len(v)-1]
	for _, e := range v {
		if m <= e {
			m = e
		}
	}
	return
}

// Sum returns the sum of all values from the given slice.
func Sum(v []float32) (s float32) {
	for _, e := range v {
		s += e
	}
	return
}

// ArgMinMax finds the indices of min and max arguments.
func ArgMinMax(v []float32) (imin, imax int) {
	if len(v) < 1 {
		return
	}
	vmin, vmax := v[0], v[0]
	imin, imax = 0, 0
	for i := 1; i < len(v); i++ {
		if v[i] < vmin {
			imin = i
			vmin = v[i]
		}
		if v[i] > vmax {
			imax = i
			vmax = v[i]
		}
	}
	return
}

// ArgMax finds the index of the max argument.
func ArgMax(v []float32) int {
	_, imax := ArgMinMax(v)
	return imax
}

// ArgMin finds the index of the min argument.
func ArgMin(v []float32) int {
	imin, _ := ArgMinMax(v)
	return imin
}

// MakeFloatMatrix returns a new 2-dimensional slice.
func MakeFloatMatrix(rows, cols int) [][]float32 {
	matrix := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float32, cols)
	}
	return matrix
}

// StrToFloatSlice parses a string representation of a slice of float32 values.
func StrToFloatSlice(str string) ([]float32, error) {
	spl := strings.Fields(str)
	data := make([]float32, len(spl))
	for i, v := range spl {
		if num, err := strconv.ParseFloat(v, 32); err == nil {
			data[i] = float32(num)
		} else {
			return nil, err
		}
	}
	return data, nil
}

// SoftMax returns the results of the softmax function.
func SoftMax(v []float32) (sm []float32) {
	c := Max(v)
	var sum float32 = 0
	for _, e := range v {
		sum += float32(math.Exp(float64(e - c)))
	}
	sm = make([]float32, len(v))
	for i, v := range v {
		sm[i] = float32(math.Exp(float64(v-c))) / sum
	}
	return sm
}

// CumSum computes the cumulative sum of src into dst, and returns dst.
func CumSum(dst, src []float32) []float32 {
	return internal.CumSum(dst, src)
}
