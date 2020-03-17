// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rnd

import (
	"golang.org/x/exp/rand"
	"math"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/mat/rnd/uniform"
	"saientist.dev/spago/pkg/utils"
)

func Bernoulli(r, c int, prob float64, source rand.Source) mat.Matrix {
	out := mat.NewEmptyDense(r, c)
	dist := uniform.New(0.0, 1.0, source)
	for i := 0; i < out.Size(); i++ {
		val := dist.Next()
		if val < prob {
			out.Set(math.Floor(val), i)
		} else {
			out.Set(math.Floor(val)+1.0, i)
		}
	}
	return out
}

func ShuffleInPlace(xs []int, source rand.Source) []int {
	swap := func(i, j int) { xs[i], xs[j] = xs[j], xs[i] }
	if source != nil {
		rand.New(source).Shuffle(len(xs), swap)
	} else {
		rand.Shuffle(len(xs), swap) // use global rand
	}
	return xs
}

// WeightedChoice performs a random generation of the indices based of the probability distribution itself.
func WeightedChoice(dist []float64) int {
	rnd := rand.Float64()
	cumulativeProb := 0.0
	for i, prob := range dist {
		cumulativeProb += prob
		if rnd < cumulativeProb {
			return i
		}
	}
	return 0
}

// GetUniqueRandomInt generates n mutually exclusive integers up to max, using the default random source.
// The callback checks whether a generated number can be accepted, or not.
func GetUniqueRandomInt(n, max int, valid func(r int) bool) []int {
	a := make([]int, n)
	for i := 0; i < n; i++ {
		r := rand.Intn(max)
		for !valid(r) || utils.ContainsInt(a, r) {
			r = rand.Intn(max)
		}
		a[i] = r
	}
	return a
}
