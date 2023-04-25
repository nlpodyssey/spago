// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matfuncs

import (
	"testing"
)

func TestAdd32(t *testing.T) {
	testAdd(t, Add32, 1e-6)
}

func TestAdd64(t *testing.T) {
	testAdd(t, Add64, 1e-6)
}

func testAdd[F Float](t *testing.T, fn func(x1, x2, y []F), eps float64) {
	t.Parallel()

	x1 := make([]F, 0, 2_000)
	x2 := make([]F, 0, 2_000)
	expected := make([]F, 0, 2_000)
	actual := make([]F, 0, 2_000)

	for size := 0; size < 2_000; size++ {
		x1 = x1[:size]
		x2 = x2[:size]
		expected = expected[:size]
		actual = actual[:size]
		RandVec(x1)
		RandVec(x2)
		testingAdd(x1, x2, expected)

		fn(x1, x2, actual)

		RequireSlicesInDelta(t, expected, actual, eps)
	}

	// Try different alignments
	x1 = x1[:16]
	x2 = x2[:16]
	expected = expected[:16]
	actual = actual[:16]
	for offset := range x1 {
		testingAdd(x1[offset:], x2[offset:], expected[offset:])
		fn(x1[offset:], x2[offset:], actual[offset:])
		RequireSlicesInDelta(t, expected[offset:], actual[offset:], eps)
	}
}

func BenchmarkAdd32(b *testing.B) {
	benchmarkAdd(b, Add32)
}

func BenchmarkAdd64(b *testing.B) {
	benchmarkAdd(b, Add64)
}

func benchmarkAdd[F Float](b *testing.B, fn func(x1, x2, y []F)) {
	size := 1_000_000
	x1 := NewRandVec[F](size)
	x2 := NewRandVec[F](size)
	y := make([]F, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(x1, x2, y)
	}
}

func testingAdd[F Float](x1, x2, y []F) {
	if len(x1) != len(x2) || len(x1) != len(y) {
		panic("len mismatch")
	}
	for i, x1v := range x1 {
		y[i] = x1v + x2[i]
	}
}
