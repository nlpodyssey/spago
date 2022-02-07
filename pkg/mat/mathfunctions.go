// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat/internal/f32/math32"
	"math"
)

// TODO: review this code once stable go 1.18 is released

// SmallestNonzero returns the smallest positive, non-zero value representable by the type.
func SmallestNonzero[T DType]() T {
	switch any(T(0)).(type) {
	case float32:
		return T(math.SmallestNonzeroFloat32)
	case float64:
		return T(math.SmallestNonzeroFloat64)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// The Pi mathematical constant.
func Pi[T DType]() T {
	return T(math.Pi)
}

// Pow returns x**y, the base-x exponential of y.
func Pow[T DType](x, y T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Pow(float32(x), float32(y)))
	case float64:
		return T(math.Pow(float64(x), float64(y)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Cos returns the cosine of the radian argument x.
func Cos[T DType](x T) T {
	return T(math.Cos(float64(x)))
}

// Sin returns the sine of the radian argument x.
func Sin[T DType](x T) T {
	return T(math.Sin(float64(x)))
}

// Cosh returns the hyperbolic cosine of x.
func Cosh[T DType](x T) T {
	return T(math.Cosh(float64(x)))
}

// Sinh returns the hyperbolic sine of x.
func Sinh[T DType](x T) T {
	return T(math.Sinh(float64(x)))
}

// Exp returns e**x, the base-e exponential of x.
func Exp[T DType](x T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Exp(float32(x)))
	case float64:
		return T(math.Exp(float64(x)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Abs returns the absolute value of x.
func Abs[T DType](x T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Abs(float32(x)))
	case float64:
		return T(math.Abs(float64(x)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Sqrt returns the square root of x.
func Sqrt[T DType](x T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Sqrt(float32(x)))
	case float64:
		return T(math.Sqrt(float64(x)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Log returns the natural logarithm of x.
func Log[T DType](x T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Log(float32(x)))
	case float64:
		return T(math.Log(float64(x)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Tan returns the tangent of the radian argument x.
func Tan[T DType](x T) T {
	return T(math.Tan(float64(x)))
}

// Tanh returns the hyperbolic tangent of x.
func Tanh[T DType](x T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Tanh(float32(x)))
	case float64:
		return T(math.Tanh(float64(x)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Max returns the larger of x or y.
func Max[T DType](x, y T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Max(float32(x), float32(y)))
	case float64:
		return T(math.Max(float64(x), float64(y)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf[T DType](sign int) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Inf(sign))
	case float64:
		return T(math.Inf(sign))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// IsInf reports whether f is an infinity, according to sign.
func IsInf[T DType](f T, sign int) bool {
	switch any(T(0)).(type) {
	case float32:
		return math32.IsInf(float32(f), sign)
	case float64:
		return math.IsInf(float64(f), sign)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN[T DType]() T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.NaN())
	case float64:
		return T(math.NaN())
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Ceil returns the least integer value greater than or equal to x.
func Ceil[T DType](x T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Ceil(float32(x)))
	case float64:
		return T(math.Ceil(float64(x)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Floor returns the greatest integer value less than or equal to x.
func Floor[T DType](x T) T {
	switch any(T(0)).(type) {
	case float32:
		return T(math32.Floor(float32(x)))
	case float64:
		return T(math.Floor(float64(x)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Round returns the nearest integer, rounding half away from zero.
func Round[T DType](x T) T {
	return T(math.Round(float64(x)))
}
