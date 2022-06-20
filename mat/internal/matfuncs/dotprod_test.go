// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matfuncs

import (
	"math/rand"
	"testing"
	"time"
)

func TestDotProd32(t *testing.T) {
	testDotProd(t, DotProd32, 1e-4)
}

func TestDotProd64(t *testing.T) {
	testDotProd(t, DotProd64, 1e-6)
}

func testDotProd[F Float](t *testing.T, fn func(x1, x2 []F) F, eps float64) {
	t.Parallel()

	rand.Seed(time.Now().Unix())

	x1 := make([]F, 0, 2_000)
	x2 := make([]F, 0, 2_000)

	for size := 0; size < 2_000; size++ {
		x1 = x1[:size]
		x2 = x2[:size]
		RandVec(x1)
		RandVec(x2)
		expected := testingDotProd(x1, x2)

		actual := fn(x1, x2)
		RequireValueInDelta(t, expected, actual, eps, "size: ", size)
	}

	// Try different alignments
	x1 = x1[:16]
	x2 = x2[:16]
	for offset := range x1 {
		expected := testingDotProd(x1[offset:], x2[offset:])
		actual := fn(x1[offset:], x2[offset:])
		RequireValueInDelta(t, expected, actual, eps, "size: ", len(x1[offset:]))
	}
}

func BenchmarkDotProd32(b *testing.B) {
	benchmarkDotProd(b, DotProd32)
}

func BenchmarkDotProd64(b *testing.B) {
	benchmarkDotProd(b, DotProd64)
}

func benchmarkDotProd[F Float](b *testing.B, fn func(x1, x2 []F) F) {
	size := 1_000_000
	x1 := NewRandVec[F](size)
	x2 := NewRandVec[F](size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fn(x1, x2)
	}
}

func testingDotProd[F Float](x1, x2 []F) (y F) {
	if len(x1) != len(x2) {
		panic("len mismatch")
	}
	for i, x1v := range x1 {
		y += x1v * x2[i]
	}
	return y
}
