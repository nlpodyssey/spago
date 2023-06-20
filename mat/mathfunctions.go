// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat/float"
)

// SmallestNonzero returns the smallest positive, non-zero value representable by the type.
func SmallestNonzero[T float.DType]() T {
	switch any(T(0)).(type) {
	case float32:
		return T(math.SmallestNonzeroFloat32)
	case float64:
		return T(math.SmallestNonzeroFloat64)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Pi mathematical constant.
func Pi[T float.DType]() T {
	return T(math.Pi)
}

// Pow returns x**y, the base-x exponential of y.
func Pow[T float.DType](x, y T) T {
	return T(math.Pow(float64(x), float64(y)))
}

// Cos returns the cosine of the radian argument x.
func Cos[T float.DType](x T) T {
	return T(math.Cos(float64(x)))
}

// Sin returns the sine of the radian argument x.
func Sin[T float.DType](x T) T {
	return T(math.Sin(float64(x)))
}

// Cosh returns the hyperbolic cosine of x.
func Cosh[T float.DType](x T) T {
	return T(math.Cosh(float64(x)))
}

// Sinh returns the hyperbolic sine of x.
func Sinh[T float.DType](x T) T {
	return T(math.Sinh(float64(x)))
}

// Exp returns e**x, the base-e exponential of x.
func Exp[T float.DType](x T) T {
	return T(math.Exp(float64(x)))
}

// Abs returns the absolute value of x.
func Abs[T float.DType](x T) T {
	return T(math.Abs(float64(x)))
}

// Sqrt returns the square root of x.
func Sqrt[T float.DType](x T) T {
	return T(math.Sqrt(float64(x)))
}

// Log returns the natural logarithm of x.
func Log[T float.DType](x T) T {
	return T(math.Log(float64(x)))
}

// Tan returns the tangent of the radian argument x.
func Tan[T float.DType](x T) T {
	return T(math.Tan(float64(x)))
}

// Tanh returns the hyperbolic tangent of x.
func Tanh[T float.DType](x T) T {
	return T(math.Tanh(float64(x)))
}

// Max returns the larger of x or y.
func Max[T float.DType](x, y T) T {
	return T(math.Max(float64(x), float64(y)))
}

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf[T float.DType](sign int) T {
	return T(math.Inf(sign))
}

// IsInf reports whether f is an infinity, according to sign.
func IsInf[T float.DType](f T, sign int) bool {
	return math.IsInf(float64(f), sign)
}

// NaN returns an IEEE 754 “not-a-number” value.
func NaN[T float.DType]() T {
	return T(math.NaN())
}

// Ceil returns the least integer value greater than or equal to x.
func Ceil[T float.DType](x T) T {
	return T(math.Ceil(float64(x)))
}

// Floor returns the greatest integer value less than or equal to x.
func Floor[T float.DType](x T) T {
	return T(math.Floor(float64(x)))
}

// Round returns the nearest integer, rounding half away from zero.
func Round[T float.DType](x T) T {
	return T(math.Round(float64(x)))
}
