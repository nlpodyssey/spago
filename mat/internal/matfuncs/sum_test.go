// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matfuncs

import (
	"math/rand"
	"testing"
	"time"
)

func TestSum32(t *testing.T) {
	testSum(t, Sum32, 1e-3)
}

func TestSum64(t *testing.T) {
	testSum(t, Sum64, 1e-6)
}

func testSum[F Float](t *testing.T, fn func(x []F) F, eps float64) {
	t.Parallel()

	rand.Seed(time.Now().Unix())

	x := make([]F, 0, 2_000)

	for size := 0; size < 2_000; size++ {
		x = x[:size]
		RandVec(x)
		expected := testingSum(x)

		actual := fn(x)
		RequireValueInDelta(t, expected, actual, eps, "size: ", size)
	}

	// Try different alignments
	x = x[:16]
	for offset := range x {
		expected := testingSum(x[offset:])
		actual := fn(x[offset:])
		RequireValueInDelta(t, expected, actual, eps, "size: ", len(x[offset:]))
	}
}

func BenchmarkSum32(b *testing.B) {
	benchmarkSum(b, Sum32)
}

func BenchmarkSum64(b *testing.B) {
	benchmarkSum(b, Sum64)
}

func benchmarkSum[F Float](b *testing.B, fn func(x []F) F) {
	size := 1_000_000
	x := NewRandVec[F](size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fn(x)
	}
}

func testingSum[F Float](x []F) (y F) {
	for _, v := range x {
		y += v
	}
	return
}
