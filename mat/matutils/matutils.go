// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matutils

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/internal/f32"
	"github.com/nlpodyssey/spago/mat/internal/f64/asm64"
	"math"
	"strconv"
	"strings"
)

// StrToFloatSlice parses a string representation of a slice of T values.
func StrToFloatSlice[T mat.DType](str string) ([]T, error) {
	var bitSize int
	switch any(T(0)).(type) {
	case float32:
		bitSize = 32
	case float64:
		bitSize = 64
	default:
		panic(fmt.Sprintf("matutils: unexpected type %T", T(0)))
	}

	spl := strings.Fields(str)
	data := make([]T, len(spl))

	for i, v := range spl {
		num, err := strconv.ParseFloat(v, bitSize)
		if err != nil {
			return nil, err
		}
		data[i] = T(num)
	}

	return data, nil
}

// SoftMax returns the results of the softmax function.
func SoftMax[T mat.DType](v []T) (sm []T) {
	c := max(v)
	var sum T = 0
	sm = make([]T, len(v))
	for _, e := range v {
		sum += T(math.Exp(float64(e - c)))
	}
	for i, v := range v {
		sm[i] = T(math.Exp(float64(v-c))) / sum
	}
	return sm
}

// CumSum computes the cumulative sum of src into dst, and returns dst.
func CumSum[T mat.DType](dst, src []T) []T {
	switch any(T(0)).(type) {
	case float32:
		return any(f32.CumSum(any(dst).([]float32), any(src).([]float32))).([]T)
	case float64:
		return any(asm64.CumSum(any(dst).([]float64), any(src).([]float64))).([]T)
	default:
		panic(fmt.Sprintf("matutils: unexpected type %T", T(0)))
	}
}

// max returns the maximum value from the given slice, which MUST NOT be empty.
func max[T mat.DType](v []T) (m T) {
	m = v[len(v)-1]
	for _, e := range v {
		if m <= e {
			m = e
		}
	}
	return
}
