/// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matfuncs

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat/internal/rand"
	"math"

	"testing"
	"time"
)

// Float is a type constraint for float32 and float64.
type Float interface {
	float32 | float64
}

var randForTesting = rand.New(rand.NewSource(uint64(time.Now().Unix())))

func RandFloat[F Float]() F {
	return F(randForTesting.NormFloat64() * 0.5)
}

// RandVec fills the given vector with random values.
func RandVec[F Float](v []F) {
	for i := range v {
		v[i] = RandFloat[F]()
	}
}

// NewRandVec creates a vector of given size, filled with random values.
func NewRandVec[F Float](size int) []F {
	v := make([]F, size)
	RandVec(v)
	return v
}

func RequireSlicesInDelta[F Float](t *testing.T, expected, actual []F, eps float64) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Fatalf("expected len %d, actual %d", len(expected), len(actual))
	}
	if len(expected) == 0 {
		return
	}
	var _ = actual[len(expected)-1]
	for i, e := range expected {
		a := actual[i]
		if d := math.Abs(float64(e - a)); d > eps {
			t.Fatalf(
				"value at index %d: expected %G ± %G, actual %G\n"+
					"slice size: %d\n"+
					"expected:   %G\n"+
					"actual:     %G",
				i, e, eps, a, len(expected), expected, actual,
			)
		}
	}
}

func RequireValueInDelta[F Float](t *testing.T, expected, actual F, eps float64, msg ...any) {
	t.Helper()
	if d := math.Abs(float64(expected - actual)); d > eps {
		t.Fatalf("expected %G ± %G, actual %G\n%s", expected, eps, actual, fmt.Sprint(msg...))
	}
}
