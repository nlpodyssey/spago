// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mattest_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/mattest"
	"github.com/stretchr/testify/assert"
)

func TestAssertMatrixEquals(t *testing.T) {
	t.Run("float32", testAssertMatrixEquals[float32])
	t.Run("float64", testAssertMatrixEquals[float64])
}

func testAssertMatrixEquals[T mat.DType](t *testing.T) {
	for _, tc := range matrixEqualsTestCases[T]() {
		t.Run(tc.name, func(t *testing.T) {
			dt := new(dummyT)
			if !tc.success {
				dt.error = func(args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprint(args...))
				}
				dt.errorf = func(format string, args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprintf(format, args...))
				}
			}

			mattest.AssertMatrixEquals(dt, tc.expected, tc.actual, tc.args...)

			assert.Positive(t, dt.helperCalls)
			assert.Equal(t, 0, dt.failNowCalls)
			if tc.success {
				assert.Equal(t, 0, dt.errorfCalls+dt.errorCalls)
			} else {
				assert.Equal(t, 1, dt.errorfCalls+dt.errorCalls)
			}
		})
	}
}

func TestRequireMatrixEquals(t *testing.T) {
	t.Run("float32", testRequireMatrixEquals[float32])
	t.Run("float64", testRequireMatrixEquals[float64])
}

func testRequireMatrixEquals[T mat.DType](t *testing.T) {
	for _, tc := range matrixEqualsTestCases[T]() {
		t.Run(tc.name, func(t *testing.T) {
			dt := new(dummyT)
			if !tc.success {
				dt.error = func(args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprint(args...))
				}
				dt.errorf = func(format string, args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprintf(format, args...))
				}
			}

			mattest.RequireMatrixEquals(dt, tc.expected, tc.actual, tc.args...)

			assert.Positive(t, dt.helperCalls)
			if tc.success {
				assert.Equal(t, 0, dt.errorfCalls+dt.errorCalls)
				assert.Equal(t, 0, dt.failNowCalls)
			} else {
				assert.Equal(t, 1, dt.errorfCalls+dt.errorCalls)
				assert.Equal(t, 1, dt.failNowCalls)
			}
		})
	}
}

func TestAssertMatrixInDelta(t *testing.T) {
	t.Run("float32", testAssertMatrixInDelta[float32])
	t.Run("float64", testAssertMatrixInDelta[float64])
}

func testAssertMatrixInDelta[T mat.DType](t *testing.T) {
	for _, tc := range matrixInDeltaTestCases[T]() {
		t.Run(tc.name, func(t *testing.T) {
			dt := new(dummyT)
			if !tc.success {
				dt.error = func(args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprint(args...))
				}
				dt.errorf = func(format string, args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprintf(format, args...))
				}
			}

			mattest.AssertMatrixInDelta(dt, tc.expected, tc.actual, tc.delta, tc.args...)

			assert.Positive(t, dt.helperCalls)
			assert.Equal(t, 0, dt.failNowCalls)
			if tc.success {
				assert.Equal(t, 0, dt.errorfCalls+dt.errorCalls)
			} else {
				assert.Equal(t, 1, dt.errorfCalls+dt.errorCalls)
			}
		})
	}
}

func TestRequireMatrixInDelta(t *testing.T) {
	t.Run("float32", testRequireMatrixInDelta[float32])
	t.Run("float64", testRequireMatrixInDelta[float64])
}

func testRequireMatrixInDelta[T mat.DType](t *testing.T) {
	for _, tc := range matrixInDeltaTestCases[T]() {
		t.Run(tc.name, func(t *testing.T) {
			dt := new(dummyT)
			if !tc.success {
				dt.error = func(args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprint(args...))
				}
				dt.errorf = func(format string, args ...any) {
					assert.Equal(t, tc.errorMsg, fmt.Sprintf(format, args...))
				}
			}

			mattest.RequireMatrixInDelta(dt, tc.expected, tc.actual, tc.delta, tc.args...)

			assert.Positive(t, dt.helperCalls)
			if tc.success {
				assert.Equal(t, 0, dt.errorfCalls+dt.errorCalls)
				assert.Equal(t, 0, dt.failNowCalls)
			} else {
				assert.Equal(t, 1, dt.errorfCalls+dt.errorCalls)
				assert.Equal(t, 1, dt.failNowCalls)
			}
		})
	}
}

type matrixEqualsTestCase struct {
	name     string
	expected mat.Matrix
	actual   mat.Matrix
	args     []any
	success  bool
	errorMsg string
}

func matrixEqualsTestCases[T mat.DType]() []matrixEqualsTestCase {
	return []matrixEqualsTestCase{
		{
			name:     "no errors",
			expected: mat.NewScalar[T](42),
			actual:   mat.NewScalar[T](42),
			args:     nil,
			success:  true,
			errorMsg: "",
		},
		{
			name:     "expected nil",
			expected: nil,
			actual:   mat.NewScalar[T](42),
			args:     nil,
			success:  false,
			errorMsg: "the expected matrix must not be nil",
		},
		{
			name:     "actual nil",
			expected: mat.NewScalar[T](42),
			actual:   nil,
			args:     nil,
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices are not equal:",
				"expected:",
				"[42]",
				"actual:",
				"nil",
				"",
			}, "\n"),
		},
		{
			name:     "actual nil - args",
			expected: mat.NewScalar[T](42),
			actual:   nil,
			args:     []any{"foo ", 123},
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices are not equal:",
				"expected:",
				"[42]",
				"actual:",
				"nil",
				"foo 123",
			}, "\n"),
		},
		{
			name:     "matrices not equal",
			expected: mat.NewScalar[T](42),
			actual:   mat.NewScalar[T](0),
			args:     nil,
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices are not equal:",
				"expected:",
				"[42]",
				"actual:",
				"[0]",
				"",
			}, "\n"),
		},
		{
			name:     "matrices not equal - args",
			expected: mat.NewScalar[T](42),
			actual:   mat.NewScalar[T](0),
			args:     []any{"foo ", 123},
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices are not equal:",
				"expected:",
				"[42]",
				"actual:",
				"[0]",
				"foo 123",
			}, "\n"),
		},
	}
}

type matrixInDeltaTestCase struct {
	name     string
	expected mat.Matrix
	actual   mat.Matrix
	delta    float64
	args     []any
	success  bool
	errorMsg string
}

func matrixInDeltaTestCases[T mat.DType]() []matrixInDeltaTestCase {
	return []matrixInDeltaTestCase{
		{
			name:     "no errors",
			expected: mat.NewScalar[T](42),
			actual:   mat.NewScalar[T](43),
			delta:    1,
			args:     nil,
			success:  true,
			errorMsg: "",
		},
		{
			name:     "expected nil",
			expected: nil,
			actual:   mat.NewScalar[T](42),
			delta:    1,
			args:     nil,
			success:  false,
			errorMsg: "the expected matrix must not be nil",
		},
		{
			name:     "actual nil",
			expected: mat.NewScalar[T](42),
			actual:   nil,
			delta:    1,
			args:     nil,
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices values are not within delta 1:",
				"expected:",
				"[42]",
				"actual:",
				"nil",
				"",
			}, "\n"),
		},
		{
			name:     "actual nil - args",
			expected: mat.NewScalar[T](42),
			actual:   nil,
			delta:    1,
			args:     []any{"foo ", 123},
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices values are not within delta 1:",
				"expected:",
				"[42]",
				"actual:",
				"nil",
				"foo 123",
			}, "\n"),
		},
		{
			name:     "matrices not equal",
			expected: mat.NewScalar[T](42),
			actual:   mat.NewScalar[T](44),
			delta:    1,
			args:     nil,
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices values are not within delta 1:",
				"expected:",
				"[42]",
				"actual:",
				"[44]",
				"",
			}, "\n"),
		},
		{
			name:     "matrices not equal - args",
			expected: mat.NewScalar[T](42),
			actual:   mat.NewScalar[T](44),
			delta:    1,
			args:     []any{"foo ", 123},
			success:  false,
			errorMsg: strings.Join([]string{
				"Matrices values are not within delta 1:",
				"expected:",
				"[42]",
				"actual:",
				"[44]",
				"foo 123",
			}, "\n"),
		},
	}
}

type dummyT struct {
	helper      func()
	helperCalls int

	error      func(args ...any)
	errorCalls int

	errorf      func(format string, args ...any)
	errorfCalls int

	failNow      func()
	failNowCalls int
}

func (t *dummyT) Helper() {
	t.helperCalls++
	if t.helper != nil {
		t.helper()
	}
}

func (t *dummyT) Error(args ...any) {
	t.errorCalls++
	if t.error != nil {
		t.error(args...)
	}
}

func (t *dummyT) Errorf(format string, args ...any) {
	t.errorfCalls++
	if t.errorf != nil {
		t.errorf(format, args...)
	}
}

func (t *dummyT) FailNow() {
	t.failNowCalls++
	if t.failNow != nil {
		t.failNow()
	}
}
