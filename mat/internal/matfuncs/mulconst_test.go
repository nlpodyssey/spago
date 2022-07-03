// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matfuncs

import (
	"math/rand"
	"testing"
	"time"
)

func TestMulConst32(t *testing.T) {
	testMulConst(t, MulConst32, 1e-6)
}

func TestMulConst64(t *testing.T) {
	testMulConst(t, MulConst64, 1e-6)
}

func testMulConst[F Float](t *testing.T, fn func(c F, x, y []F), eps float64) {
	t.Parallel()

	rand.Seed(time.Now().Unix())

	x := make([]F, 0, 2_000)
	expected := make([]F, 0, 2_000)
	actual := make([]F, 0, 2_000)

	for size := 0; size < 2_000; size++ {
		c := RandFloat[F]()
		x = x[:size]
		expected = expected[:size]
		actual = actual[:size]
		RandVec(x)
		testingMulConst(c, x, expected)

		fn(c, x, actual)

		RequireSlicesInDelta(t, expected, actual, eps)
	}

	// Try different alignments
	x = x[:16]
	expected = expected[:16]
	actual = actual[:16]
	for offset := range x {
		c := RandFloat[F]()
		testingMulConst(c, x[offset:], expected[offset:])
		fn(c, x[offset:], actual[offset:])
		RequireSlicesInDelta(t, expected[offset:], actual[offset:], eps)
	}
}

func BenchmarkMulConst32(b *testing.B) {
	benchmarkMulConst(b, MulConst32)
}

func BenchmarkMulConst64(b *testing.B) {
	benchmarkMulConst(b, MulConst64)
}

func benchmarkMulConst[F Float](b *testing.B, fn func(c F, x, y []F)) {
	size := 1_000_000
	c := RandFloat[F]()
	x := NewRandVec[F](size)
	y := make([]F, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(c, x, y)
	}
}

func testingMulConst[F Float](c F, x, y []F) {
	if len(x) != len(y) {
		panic("len mismatch")
	}
	for i, x1v := range x {
		y[i] = x1v * c
	}
}
