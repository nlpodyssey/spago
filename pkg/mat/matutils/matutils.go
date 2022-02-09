// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matutils

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/internal/f32"
	"github.com/nlpodyssey/spago/pkg/mat/internal/f32/math32"
	"github.com/nlpodyssey/spago/pkg/mat/internal/f64/asm64"
	"math"
	"strconv"
	"strings"
)

// Max returns the maximum value from the given slice, which MUST NOT be empty.
func Max[T mat.DType](v []T) (m T) {
	m = v[len(v)-1]
	for _, e := range v {
		if m <= e {
			m = e
		}
	}
	return
}

// Sum returns the sum of all values from the given slice.
func Sum[T mat.DType](v []T) (s T) {
	for _, e := range v {
		s += e
	}
	return
}

// ArgMinMax finds the indices of min and max arguments.
func ArgMinMax[T mat.DType](v []T) (imin, imax int) {
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
func ArgMax[T mat.DType](v []T) int {
	_, imax := ArgMinMax(v)
	return imax
}

// StrToFloatSlice parses a string representation of a slice of T values.
func StrToFloatSlice[T mat.DType](str string) ([]T, error) {
	spl := strings.Fields(str)
	data := make([]T, len(spl))

	switch any(T(0)).(type) {
	case float32:
		for i, v := range spl {
			num, err := strconv.ParseFloat(v, 32)
			if err != nil {
				return nil, err
			}
			data[i] = T(num)
		}
	case float64:
		for i, v := range spl {
			num, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, err
			}
			data[i] = T(num)
		}
	default:
		panic(fmt.Sprintf("matutils: unexpected type %T", T(0)))
	}

	return data, nil
}

// SoftMax returns the results of the softmax function.
func SoftMax[T mat.DType](v []T) (sm []T) {
	c := Max(v)
	var sum T = 0
	sm = make([]T, len(v))

	switch any(T(0)).(type) {
	case float32:
		for _, e := range v {
			sum += T(math32.Exp(float32(e - c)))
		}
		for i, v := range v {
			sm[i] = T(math32.Exp(float32(v-c))) / sum
		}
	case float64:
		for _, e := range v {
			sum += T(math.Exp(float64(e - c)))
		}
		for i, v := range v {
			sm[i] = T(math.Exp(float64(v-c))) / sum
		}
	default:
		panic(fmt.Sprintf("matutils: unexpected type %T", T(0)))
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
