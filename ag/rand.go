// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"time"

	"github.com/nlpodyssey/spago/mat/rand"
)

var globalGenerator = rand.NewLockedRand(12345)

// Seed sets the seed for generating random numbers to the current time (converted to uint64).
func Seed() *rand.LockedRand {
	globalGenerator.Seed(uint64(time.Now().UnixNano()))
	return globalGenerator
}

// ManualSeed sets the seed for generating random numbers.
func ManualSeed(seed uint64) *rand.LockedRand {
	globalGenerator.Seed(seed)
	return globalGenerator
}

// Rand returns the global random number generator.
func Rand() *rand.LockedRand {
	return globalGenerator
}
