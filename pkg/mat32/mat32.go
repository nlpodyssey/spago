// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"github.com/nlpodyssey/spago/pkg/mat32/internal/math32"
	"math"
)

// Float is the main float type for the mat32 package. It is an alias for float32.
type Float = float32

const (
	// SmallestNonzeroFloat corresponds to math.SmallestNonzeroFloat32.
	SmallestNonzeroFloat Float = math.SmallestNonzeroFloat32
	// Pi mathematical constant.
	Pi Float = math.Pi
)

// Pow returns x**y, the base-x exponential of y.
func Pow(x, y Float) Float {
	return math32.Pow(x, y)
}

// Cos returns the cosine of the radian argument x.
func Cos(x Float) Float {
	return Float(math.Cos(float64(x)))
}

// Sin returns the sine of the radian argument x.
func Sin(x Float) Float {
	return Float(math.Sin(float64(x)))
}

// Cosh returns the hyperbolic cosine of x.
func Cosh(x Float) Float {
	return Float(math.Cosh(float64(x)))
}

// Sinh returns the hyperbolic sine of x.
func Sinh(x Float) Float {
	return Float(math.Sinh(float64(x)))
}

// Exp returns e**x, the base-e exponential of x.
func Exp(x Float) Float {
	return math32.Exp(x)
}

// Abs returns the absolute value of x.
func Abs(x Float) Float {
	return math32.Abs(x)
}

// Sqrt returns the square root of x.
func Sqrt(x Float) Float {
	return math32.Sqrt(x)
}

// Log returns the natural logarithm of x.
func Log(x Float) Float {
	return math32.Log(x)
}

// Tan returns the tangent of the radian argument x.
func Tan(x Float) Float {
	return Float(math.Tan(float64(x)))
}

// Tanh returns the hyperbolic tangent of x.
func Tanh(x Float) Float {
	return math32.Tanh(x)
}

// Max returns the larger of x or y.
func Max(x, y Float) Float {
	return math32.Max(x, y)
}

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) Float {
	return math32.Inf(sign)
}

// IsInf reports whether f is an infinity, according to sign.
func IsInf(f Float, sign int) bool {
	return math32.IsInf(f, sign)
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() Float {
	return math32.NaN()
}

// Ceil returns the least integer value greater than or equal to x.
func Ceil(x Float) Float {
	return math32.Ceil(x)
}

// Floor returns the greatest integer value less than or equal to x.
func Floor(x Float) Float {
	return math32.Floor(x)
}

// Round returns the nearest integer, rounding half away from zero.
func Round(x Float) Float {
	return Float(math.Round(float64(x)))
}
