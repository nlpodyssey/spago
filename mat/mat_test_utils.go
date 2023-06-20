// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mattest provides utilities for testing code involving spaGO matrices.
package mat

import (
	"fmt"
)

// T requires a subset of methods from testing.TB.
// This interface is primarily useful to simplify the testing of the
// package itself.
type T interface {
	Helper()
	Error(args ...any)
	Errorf(format string, args ...any)
	FailNow()
}

// AssertMatrixEquals tests whether expected is equal to actual; if not,
// T.Error or T.Errorf are called, providing useful information.
//
// It returns whether the comparison succeeded.
//
// The expected matrix is not allowed to be nil, otherwise the function always
// produces an error.
func AssertMatrixEquals(t T, expected, actual Tensor, args ...any) bool {
	t.Helper()
	if expected == nil {
		t.Error("the expected matrix must not be nil")
		return false
	}
	if actual == nil {
		t.Errorf("Matrices are not equal:\nexpected:\n%g\nactual:\nnil\n%s",
			expected, fmt.Sprint(args...))
		return false
	}
	if !Equal(expected, actual) {
		t.Errorf("Matrices are not equal:\nexpected:\n%g\nactual:\n%g\n%s",
			expected, actual, fmt.Sprint(args...))
		return false
	}
	return true
}

// RequireMatrixEquals tests whether expected is equal to actual; if not,
// T.Error or T.Errorf are called, providing useful information, followed by
// T.FailNow.
//
// The expected matrix is not allowed to be nil, otherwise the function always
// produces an error and fails.
func RequireMatrixEquals(t T, expected, actual Matrix, args ...any) {
	t.Helper()
	if !AssertMatrixEquals(t, expected, actual, args...) {
		t.FailNow()
	}
}

// AssertMatrixInDelta tests whether expected and actual have the same shape
// and all elements at the same positions are within delta; if not, T.Error
// or T.Errorf are called, providing useful information.
//
// It returns whether the comparison succeeded.
//
// The expected matrix is not allowed to be nil, otherwise the function always
// produces an error.
func AssertMatrixInDelta(t T, expected, actual Matrix, delta float64, args ...any) bool {
	t.Helper()
	if expected == nil {
		t.Error("the expected matrix must not be nil")
		return false
	}
	if actual == nil {
		t.Errorf("Matrices values are not within delta %g:\nexpected:\n%g\nactual:\nnil\n%s",
			delta, expected, fmt.Sprint(args...))
		return false
	}
	if !InDelta(expected, actual, delta) {
		t.Errorf("Matrices values are not within delta %g:\nexpected:\n%g\nactual:\n%g\n%s",
			delta, expected, actual, fmt.Sprint(args...))
		return false
	}
	return true
}

// RequireMatrixInDelta tests whether expected and actual have the same shape
// and all elements at the same positions are within delta; if not, T.Error
// or T.Errorf are called, providing useful information, followed by T.FailNow.
//
// The expected matrix is not allowed to be nil, otherwise the function always
// produces an error and fails.
func RequireMatrixInDelta(t T, expected, actual Matrix, delta float64, args ...any) {
	t.Helper()
	if !AssertMatrixInDelta(t, expected, actual, delta, args...) {
		t.FailNow()
	}
}
