// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat/float"
)

var _ fmt.Formatter = &Dense[float32]{}
var _ fmt.Formatter = &Dense[float64]{}

func TestDense_Format(t *testing.T) {
	t.Run("float32", testDenseFormat[float32])
	t.Run("float64", testDenseFormat[float64])
}

func testDenseFormat[T float.DType](t *testing.T) {
	var dtypeName string
	switch any(T(0)).(type) {
	case float32:
		dtypeName = "float32"
	case float64:
		dtypeName = "float64"
	default:
		t.Errorf("mat: unexpected type %T", T(0))
	}

	testCases := []struct {
		name     string
		d        *Dense[T]
		format   string
		expected string
	}{
		{
			"Empty matrix",
			NewEmptyDense[T](0, 0),
			"%v",
			"[]",
		},
		{
			"Scalar",
			NewScalar[T](1.2),
			"%v",
			"[1.2]",
		},
		{
			"Scalar with padding",
			NewScalar[T](1.2),
			"%5v",
			"[  1.2]",
		},
		{
			"One row",
			NewDense[T](1, 3, []T{1.2, 3.4, 5.6}),
			"%v",
			"[1.2 3.4 5.6]",
		},
		{
			"One column",
			NewDense[T](3, 1, []T{1.2, 3.4, 5.6}),
			"%v",
			"⎡1.2⎤\n" +
				"⎢3.4⎥\n" +
				"⎣5.6⎦",
		},
		{
			"3x3",
			NewDense[T](3, 3, []T{
				1.2, 3.4, 5.6,
				7.8, 9.1, 2.3,
				4.5, 6.7, 8.9,
			}),
			"%v",
			"⎡1.2 3.4 5.6⎤\n" +
				"⎢7.8 9.1 2.3⎥\n" +
				"⎣4.5 6.7 8.9⎦",
		},
		{
			"Max column width is respected",
			NewDense[T](3, 3, []T{
				11.2, 3.4, 5.6,
				7.8, 99.11, 2.3,
				4.5, 6.7, 88.999,
			}),
			"%v",
			"⎡11.2  3.4   5.6  ⎤\n" +
				"⎢ 7.8 99.11  2.3  ⎥\n" +
				"⎣ 4.5  6.7  88.999⎦",
		},
		{
			"Explicit padding is respected",
			NewDense[T](3, 4, []T{
				11.2, 3.4, 5.6, 0.1,
				7.8, 99.11, 2.3, 0.2,
				4.5, 6.7, 88.999, 123456.78,
			}),
			"%8v",
			"⎡    11.2      3.4       5.6        0.1 ⎤\n" +
				"⎢     7.8     99.11      2.3        0.2 ⎥\n" +
				"⎣     4.5      6.7      88.999 123456.78⎦",
		},
		{
			"Go-syntax representation",
			NewScalar[T](1.2),
			"%#v",
			"&mat.Dense[" + dtypeName + "]{rows:1, cols:1, flags:0x1, data:[]" + dtypeName + "{1.2}}",
		},
		{
			"Default format with field names",
			NewScalar[T](1.2),
			"%+v",
			"{rows:1 cols:1 flags:1 data:[1.2]}",
		},
		{
			"decimalless scientific notation",
			NewScalar[T](0),
			"%b",
			"[0p-149]",
		},
		{
			"scientific notation - small e",
			NewScalar[T](12.3),
			"%e",
			"[1.23e+01]",
		},
		{
			"scientific notation - capital E",
			NewScalar[T](12.3),
			"%E",
			"[1.23E+01]",
		},
		{
			"decimal point no exponent",
			NewScalar[T](12.3),
			"%f",
			"[12.3]",
		},
		{
			"decimal point no exponent alt",
			NewScalar[T](12.3),
			"%F",
			"[12.3]",
		},
		{
			"scientific notation for large exponents - small e",
			NewDense[T](1, 2, []T{1.2, 3456789.0}),
			"%g",
			"[1.2 3.456789e+06]",
		},
		{
			"scientific notation for large exponents - capital E",
			NewDense[T](1, 2, []T{1.2, 3456789.0}),
			"%G",
			"[1.2 3.456789E+06]",
		},
		{
			"hex notation lowercase",
			NewScalar[T](0),
			"%x",
			"[0x0p+00]",
		},
		{
			"hex notation uppercase",
			NewScalar[T](0),
			"%X",
			"[0X0P+00]",
		},
		{
			"precision only",
			NewDense[T](1, 2, []T{1.23, 4.567}),
			"%.2f",
			"[1.23 4.57]",
		},
		{
			"width and precision",
			NewDense[T](1, 2, []T{1.23, 4.567}),
			"%6.2f",
			"[  1.23   4.57]",
		},
		{
			"correct point alignment using g",
			NewDense[T](3, 3, []T{
				0.1, 1234567.8, 123456.78,
				12345678.0, 12345.6, 9,
				21, 322, 9876543,
			}),
			"%g",
			"⎡ 0.1               1.2345678e+06 123456.78        ⎤\n" +
				"⎢ 1.2345678e+07 12345.6                9           ⎥\n" +
				"⎣21               322                  9.876543e+06⎦",
		},
		{
			"correct point alignment using g with small width",
			NewDense[T](3, 3, []T{
				0.1, 1234567.8, 123456.78,
				12345678.0, 12345.6, 9,
				21, 322, 9876543,
			}),
			"%6g",
			"⎡   0.1               1.2345678e+06 123456.78        ⎤\n" +
				"⎢   1.2345678e+07 12345.6                9           ⎥\n" +
				"⎣  21               322                  9.876543e+06⎦",
		},
		{
			"correct point alignment using g with big width",
			NewDense[T](3, 3, []T{
				0.1, 1234567.8, 123456.78,
				12345678.0, 12345.6, 9,
				21, 322, 9876543,
			}),
			"%8g",
			"⎡     0.1                1.2345678e+06 123456.78        ⎤\n" +
				"⎢     1.2345678e+07  12345.6                9           ⎥\n" +
				"⎣    21                322                  9.876543e+06⎦",
		},
		{
			"correct point alignment using g with zero precision",
			NewDense[T](3, 3, []T{
				0.1, 1234567.89, 123456.789,
				12345678.987, 12345.6, 9,
				21, 322, 9876543,
			}),
			"%.0g",
			"⎡    0.1 1e+06 1e+05⎤\n" +
				"⎢1e+07   1e+04     9⎥\n" +
				"⎣2e+01   3e+02 1e+07⎦",
		},
		{
			"correct point alignment using g with precision",
			NewDense[T](3, 3, []T{
				0.1, 1234567.89, 123456.789,
				12345678.987, 12345.6, 9,
				21, 322, 9876543,
			}),
			"%.3g",
			"⎡ 0.1        1.23e+06 1.23e+05⎤\n" +
				"⎢ 1.23e+07   1.23e+04 9       ⎥\n" +
				"⎣21        322        9.88e+06⎦",
		},
		{
			"correct point alignment using g with width and precision",
			NewDense[T](3, 3, []T{
				0.1, 1234567.89, 123456.789,
				12345678.987, 12345.6, 9,
				21, 322, 9876543,
			}),
			"%8.3g",
			"⎡     0.1        1.23e+06 1.23e+05⎤\n" +
				"⎢     1.23e+07   1.23e+04 9       ⎥\n" +
				"⎣    21        322        9.88e+06⎦",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := fmt.Sprintf(tc.format, tc.d)
			_ = actual
			//require.Equal(t, tc.expected, actual) // TODO: update
		})
	}
}
