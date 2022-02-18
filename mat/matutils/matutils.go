// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matutils

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/internal/f32"
	"github.com/nlpodyssey/spago/mat/internal/f64/asm64"
)

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
