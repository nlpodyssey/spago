// Copyright ©2016-2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import "github.com/nlpodyssey/spago/pkg/mat32/internal/asm/f32"

// AddConst is
//  for i := range x {
//  	x[i] += alpha
//  }
func AddConst(alpha float32, x []float32) {
	for i := range x {
		x[i] += alpha
	}
}

// DivTo is
//  for i, v := range s {
//  	dst[i] = v / t[i]
//  }
//  return dst
func DivTo(dst, s, t []float32) []float32 {
	for i, v := range s {
		dst[i] = v / t[i]
	}
	return dst
}

// Sum is
//  var sum float32
//  for i := range x {
//      sum += x[i]
//  }
func Sum(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v
	}
	return sum
}

// CumSum is
//  if len(s) == 0 {
//  	return dst
//  }
//  dst[0] = s[0]
//  for i, v := range s[1:] {
//  	dst[i+1] = dst[i] + v
//  }
//  return dst
func CumSum(dst, s []float32) []float32 {
	if len(s) == 0 {
		return dst
	}
	dst[0] = s[0]
	for i, v := range s[1:] {
		dst[i+1] = dst[i] + v
	}
	return dst
}

// GemvT computes
//  y = alpha * Aᵀ * x + beta * y
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func GemvT(m, n uintptr, alpha float32, a []float32, lda uintptr, x []float32, incX uintptr, beta float32, y []float32, incY uintptr) {
	var kx, ky, i uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(n-1) * int(incY))
	}
	switch {
	case beta == 0: // beta == 0 is special-cased to memclear
		if incY == 1 {
			for i := range y {
				y[i] = 0
			}
		} else {
			iy := ky
			for i := 0; i < int(n); i++ {
				y[iy] = 0
				iy += incY
			}
		}
	case int(incY) < 0:
		f32.ScalInc(beta, y, n, uintptr(int(-incY)))
	case incY == 1:
		f32.ScalUnitary(beta, y[:n])
	default:
		f32.ScalInc(beta, y, n, incY)
	}

	if incX == 1 && incY == 1 {
		for i = 0; i < m; i++ {
			f32.AxpyUnitaryTo(y, alpha*x[i], a[lda*i:lda*i+n], y)
		}
		return
	}
	ix := kx
	for i = 0; i < m; i++ {
		f32.AxpyInc(alpha*x[ix], a[lda*i:lda*i+n], y, n, 1, incY, 0, ky)
		ix += incX
	}
}
