// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matfuncs

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestExp32(t *testing.T) {
	testExp(t, Exp32, 1e-6)
}

func TestExp64(t *testing.T) {
	testExp(t, Exp64, 1e-6)
}

func testExp[F Float](t *testing.T, fn func(x, y []F), eps float64) {
	t.Parallel()

	rand.Seed(time.Now().Unix())

	x := make([]F, 0, 2_000)
	expected := make([]F, 0, 2_000)
	actual := make([]F, 0, 2_000)

	for size := 0; size < 2_000; size++ {
		x = x[:size]
		expected = expected[:size]
		actual = actual[:size]
		RandVec(x)
		testingExp(x, expected)

		fn(x, actual)

		RequireSlicesInDelta(t, expected, actual, eps)
	}

	// Try different alignments
	x = x[:16]
	expected = expected[:16]
	actual = actual[:16]
	for offset := range x {
		testingExp(x[offset:], expected[offset:])
		fn(x[offset:], actual[offset:])
		RequireSlicesInDelta(t, expected[offset:], actual[offset:], eps)
	}
}

func BenchmarkExp32(b *testing.B) {
	benchmarkExp(b, Exp32)
}

func BenchmarkExp64(b *testing.B) {
	benchmarkExp(b, Exp64)
}

func benchmarkExp[F Float](b *testing.B, fn func(x, y []F)) {
	size := 100_000
	x1 := NewRandVec[F](size)
	y := make([]F, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(x1, y)
	}
}

func testingExp[F Float](x, y []F) {
	if len(x) != len(y) {
		panic("len mismatch")
	}
	for i, xv := range x {
		y[i] = F(math.Exp(float64(xv)))
	}
}
