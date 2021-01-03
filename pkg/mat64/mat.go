// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import "math"

// Float is the main float type for the mat64 package. It is an alias for float64.
type Float = float64

const (
	// SmallestNonzeroFloat corresponds to math.SmallestNonzeroFloat64.
	SmallestNonzeroFloat Float = math.SmallestNonzeroFloat64
	// Pi mathematical constant.
	Pi Float = math.Pi
)

// Pow returns x**y, the base-x exponential of y.
func Pow(x, y Float) Float {
	return math.Pow(x, y)
}

// Cos returns the cosine of the radian argument x.
func Cos(x Float) Float {
	return math.Cos(x)
}

// Sin returns the sine of the radian argument x.
func Sin(x Float) Float {
	return math.Sin(x)
}

// Cosh returns the hyperbolic cosine of x.
func Cosh(x Float) Float {
	return math.Cosh(x)
}

// Sinh returns the hyperbolic sine of x.
func Sinh(x Float) Float {
	return math.Sinh(x)
}

// Exp returns e**x, the base-e exponential of x.
func Exp(x Float) Float {
	return math.Exp(x)
}

// Abs returns the absolute value of x.
func Abs(x Float) Float {
	return math.Abs(x)
}

// Sqrt returns the square root of x.
func Sqrt(x Float) Float {
	return math.Sqrt(x)
}

// Log returns the natural logarithm of x.
func Log(x Float) Float {
	return math.Log(x)
}

// Tan returns the tangent of the radian argument x.
func Tan(x Float) Float {
	return math.Tan(x)
}

// Tanh returns the hyperbolic tangent of x.
func Tanh(x Float) Float {
	return math.Tanh(x)
}

// Max returns the larger of x or y.
func Max(x, y Float) Float {
	return math.Max(x, y)
}

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) Float {
	return math.Inf(sign)
}

// IsInf reports whether f is an infinity, according to sign.
func IsInf(f Float, sign int) bool {
	return math.IsInf(f, sign)
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() Float {
	return math.NaN()
}

// Ceil returns the least integer value greater than or equal to x.
func Ceil(x Float) Float {
	return math.Ceil(x)
}

// Floor returns the greatest integer value less than or equal to x.
func Floor(x Float) Float {
	return math.Floor(x)
}

// Round returns the nearest integer, rounding half away from zero.
func Round(x Float) Float {
	return math.Round(x)
}
