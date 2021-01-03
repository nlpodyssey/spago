// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uniform

import (
	"github.com/nlpodyssey/spago/pkg/mat64/rand"
)

// Uniform is a source of uniformly distributed random numbers.
// See: https://en.wikipedia.org/wiki/Continuous_uniform_distribution.
type Uniform struct {
	Min       float64
	Max       float64
	generator *rand.LockedRand
}

// New returns a new Normal, initialized with the given min and max parameters.
func New(min, max float64, generator *rand.LockedRand) *Uniform {
	return &Uniform{
		Min:       min,
		Max:       max,
		generator: generator,
	}
}

// Next returns a random sample drawn from the distribution.
func (u Uniform) Next() float64 {
	return u.generator.Float64()*(u.Max-u.Min) + u.Min
}
