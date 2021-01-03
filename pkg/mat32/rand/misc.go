// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"github.com/nlpodyssey/spago/pkg/utils"
	"golang.org/x/exp/rand"
)

// ShuffleInPlace pseudo-randomizes the order of elements, modifying the
// given slice in-place.
func ShuffleInPlace(xs []int, generator *LockedRand) []int {
	swap := func(i, j int) { xs[i], xs[j] = xs[j], xs[i] }
	if generator != nil {
		generator.Shuffle(len(xs), swap)
	} else {
		rand.Shuffle(len(xs), swap) // Warning: use global rand
	}
	return xs
}

// WeightedChoice performs a random generation of the indices based of the probability distribution itself.
// Please note that it uses the global random.
func WeightedChoice(dist []float32) int {
	rnd := rand.Float32() // // Warning: use global rand
	var cumulativeProb float32 = 0.0
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
		r := rand.Intn(max) // Warning: use global rand
		for !valid(r) || utils.ContainsInt(a, r) {
			r = rand.Intn(max) // Warning: use global rand
		}
		a[i] = r
	}
	return a
}

// GetUniqueRandomIndices select n mutually exclusive indices, using the global random.
// The callback checks whether an extracted index can be accepted, or not.
func GetUniqueRandomIndices(n int, indices []int, valid func(r int) bool) []int {
	a := make([]int, n)
	for i := 0; i < len(a); i++ {
		r := ShuffleInPlace(indices, nil)[0] // Warning: use global rand
		for !valid(r) || utils.ContainsInt(a, r) {
			r = ShuffleInPlace(indices, nil)[0] // Warning: use global rand
		}
		a[i] = r
	}
	return a
}
