// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matfuncs

import (
	"math/rand"
	"testing"
	"time"
)

func TestAddConst32(t *testing.T) {
	testAddConst(t, AddConst32, 1e-6)
}

func TestAddConst64(t *testing.T) {
	testAddConst(t, AddConst64, 1e-6)
}

func testAddConst[F Float](t *testing.T, fn func(c F, x, y []F), eps float64) {
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
		testingAddConst(c, x, expected)

		fn(c, x, actual)

		RequireSlicesInDelta(t, expected, actual, eps)
	}

	// Try different alignments
	x = x[:16]
	expected = expected[:16]
	actual = actual[:16]
	for offset := range x {
		c := RandFloat[F]()
		testingAddConst(c, x[offset:], expected[offset:])
		fn(c, x[offset:], actual[offset:])
		RequireSlicesInDelta(t, expected[offset:], actual[offset:], eps)
	}
}

func BenchmarkAddConst32(b *testing.B) {
	benchmarkAddConst(b, AddConst32)
}

func BenchmarkAddConst64(b *testing.B) {
	benchmarkAddConst(b, AddConst64)
}

func benchmarkAddConst[F Float](b *testing.B, fn func(c F, x, y []F)) {
	size := 1_000_000
	c := RandFloat[F]()
	x := NewRandVec[F](size)
	y := make([]F, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(c, x, y)
	}
}

func testingAddConst[F Float](c F, x, y []F) {
	if len(x) != len(y) {
		panic("len mismatch")
	}
	for i, x1v := range x {
		y[i] = x1v + c
	}
}
