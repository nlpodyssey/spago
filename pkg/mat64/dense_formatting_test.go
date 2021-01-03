// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"fmt"
	"testing"
)

func TestDense_Format(t *testing.T) {
	run := func(name string, d *Dense, format, expected string) {
		t.Run(name, func(t *testing.T) {
			actual := fmt.Sprintf(format, d)
			if actual != expected {
				t.Errorf("Expected:\n%s\nActual:\n%s", expected, actual)
			}
		})
	}

	run("Empty matrix", NewEmptyDense(0, 0), "%v", "[]")

	run("Scalar", NewScalar(1.2), "%v", "[1.2]")

	run("Scalar with padding", NewScalar(1.2), "%5v", "[  1.2]")

	run("One row", NewDense(1, 3, []Float{1.2, 3.4, 5.6}), "%v",
		"[1.2 3.4 5.6]")

	run("One column", NewDense(3, 1, []Float{1.2, 3.4, 5.6}), "%v",
		""+
			"⎡1.2⎤\n"+
			"⎢3.4⎥\n"+
			"⎣5.6⎦")

	run("3x3",
		NewDense(3, 3, []Float{
			1.2, 3.4, 5.6,
			7.8, 9.1, 2.3,
			4.5, 6.7, 8.9,
		}), "%v",
		""+
			"⎡1.2 3.4 5.6⎤\n"+
			"⎢7.8 9.1 2.3⎥\n"+
			"⎣4.5 6.7 8.9⎦")

	run("Max column width is respected",
		NewDense(3, 3, []Float{
			11.2, 3.4, 5.6,
			7.8, 99.11, 2.3,
			4.5, 6.7, 88.999,
		}), "%v",
		""+
			"⎡11.2  3.4   5.6  ⎤\n"+
			"⎢ 7.8 99.11  2.3  ⎥\n"+
			"⎣ 4.5  6.7  88.999⎦")

	run("Explicit padding is respected",
		NewDense(3, 4, []Float{
			11.2, 3.4, 5.6, 0.1,
			7.8, 99.11, 2.3, 0.2,
			4.5, 6.7, 88.999, 123456.78,
		}), "%8v",
		""+
			"⎡    11.2      3.4       5.6        0.1 ⎤\n"+
			"⎢     7.8     99.11      2.3        0.2 ⎥\n"+
			"⎣     4.5      6.7      88.999 123456.78⎦")

	run("Go-syntax representation", NewScalar(1.2), "%#v",
		"&mat64.Dense{rows:1, cols:1, size:1, data:[]float64{1.2}, "+
			"viewOf:(*mat64.Dense)(nil), fromPool:true}")

	run("Default format with field names", NewScalar(1.2), "%+v",
		"{rows:1 cols:1 size:1 data:[1.2] viewOf:<nil> fromPool:true}")

	run("decimalless scientific notation", NewScalar(0), "%b", "[0p-1074]")

	run("scientific notation - small e", NewScalar(12.3), "%e", "[1.23e+01]")

	run("scientific notation - capital E", NewScalar(12.3), "%E", "[1.23E+01]")

	run("decimal point no exponent", NewScalar(12.3), "%f", "[12.3]")

	run("decimal point no exponent alt", NewScalar(12.3), "%F", "[12.3]")

	run("scientific notation for large exponents - small e",
		NewDense(1, 2, []Float{1.2, 3456789.1}), "%g", "[1.2 3.4567891e+06]")

	run("scientific notation for large exponents - capital E",
		NewDense(1, 2, []Float{1.2, 3456789.1}), "%G", "[1.2 3.4567891E+06]")

	run("hex notation lowercase", NewScalar(0), "%x", "[0x0p+00]")

	run("hex notation uppercase", NewScalar(0), "%X", "[0X0P+00]")

	run("precision only",
		NewDense(1, 2, []Float{1.23, 4.567}), "%.2f", "[1.23 4.57]")

	run("width and precision",
		NewDense(1, 2, []Float{1.23, 4.567}), "%6.2f", "[  1.23   4.57]")

	run("correct point alignment using g",
		NewDense(3, 3, []Float{
			0.1, 1234567.89, 123456.789,
			12345678.987, 12345.6, 9,
			21, 322, 9876543,
		}), "%g",
		""+
			"⎡ 0.1                  1.23456789e+06 123456.789       ⎤\n"+
			"⎢ 1.2345678987e+07 12345.6                 9           ⎥\n"+
			"⎣21                  322                   9.876543e+06⎦")

	run("correct point alignment using g with small width",
		NewDense(3, 3, []Float{
			0.1, 1234567.89, 123456.789,
			12345678.987, 12345.6, 9,
			21, 322, 9876543,
		}), "%6g",
		""+
			"⎡   0.1                  1.23456789e+06 123456.789       ⎤\n"+
			"⎢   1.2345678987e+07 12345.6                 9           ⎥\n"+
			"⎣  21                  322                   9.876543e+06⎦")

	run("correct point alignment using g with big width",
		NewDense(3, 3, []Float{
			0.1, 1234567.89, 123456.789,
			12345678.987, 12345.6, 9,
			21, 322, 9876543,
		}), "%8g",
		""+
			"⎡     0.1                   1.23456789e+06 123456.789       ⎤\n"+
			"⎢     1.2345678987e+07  12345.6                 9           ⎥\n"+
			"⎣    21                   322                   9.876543e+06⎦")

	run("correct point alignment using g with zero precision",
		NewDense(3, 3, []Float{
			0.1, 1234567.89, 123456.789,
			12345678.987, 12345.6, 9,
			21, 322, 9876543,
		}), "%.0g",
		""+
			"⎡    0.1 1e+06 1e+05⎤\n"+
			"⎢1e+07   1e+04     9⎥\n"+
			"⎣2e+01   3e+02 1e+07⎦")

	run("correct point alignment using g with precision",
		NewDense(3, 3, []Float{
			0.1, 1234567.89, 123456.789,
			12345678.987, 12345.6, 9,
			21, 322, 9876543,
		}), "%.3g",
		""+
			"⎡ 0.1        1.23e+06 1.23e+05⎤\n"+
			"⎢ 1.23e+07   1.23e+04 9       ⎥\n"+
			"⎣21        322        9.88e+06⎦")

	run("correct point alignment using g with width and precision",
		NewDense(3, 3, []Float{
			0.1, 1234567.89, 123456.789,
			12345678.987, 12345.6, 9,
			21, 322, 9876543,
		}), "%8.3g",
		""+
			"⎡     0.1        1.23e+06 1.23e+05⎤\n"+
			"⎢     1.23e+07   1.23e+04 9       ⎥\n"+
			"⎣    21        322        9.88e+06⎦")
}
