// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import "math"

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
	return Float(math.Pow(float64(x), float64(y)))
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
	return Float(math.Exp(float64(x)))
}

// Abs returns the absolute value of x.
func Abs(x Float) Float {
	return Float(math.Abs(float64(x)))
}

// Sqrt returns the square root of x.
func Sqrt(x Float) Float {
	return Float(math.Sqrt(float64(x)))
}

// Log returns the natural logarithm of x.
func Log(x Float) Float {
	return Float(math.Log(float64(x)))
}

// Tan returns the tangent of the radian argument x.
func Tan(x Float) Float {
	return Float(math.Tan(float64(x)))
}

// Tanh returns the hyperbolic tangent of x.
func Tanh(x Float) Float {
	return Float(math.Tanh(float64(x)))
}

// Max returns the larger of x or y.
func Max(x, y Float) Float {
	return Float(math.Max(float64(x), float64(y)))
}

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) Float {
	return Float(math.Inf(sign))
}

// IsInf reports whether f is an infinity, according to sign.
func IsInf(f Float, sign int) bool {
	return math.IsInf(float64(f), sign)
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() Float {
	return Float(math.NaN())
}

// Ceil returns the least integer value greater than or equal to x.
func Ceil(x Float) Float {
	return Float(math.Ceil(float64(x)))
}

// Floor returns the greatest integer value less than or equal to x.
func Floor(x Float) Float {
	return Float(math.Floor(float64(x)))
}

// Round returns the nearest integer, rounding half away from zero.
func Round(x Float) Float {
	return Float(math.Round(float64(x)))
}
