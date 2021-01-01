// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uniform

import (
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
)

// Uniform is a source of uniformly distributed random numbers.
// See: https://en.wikipedia.org/wiki/Continuous_uniform_distribution.
type Uniform struct {
	Min       float32
	Max       float32
	generator *rand.LockedRand
}

// New returns a new Normal, initialized with the given min and max parameters.
func New(min, max float32, generator *rand.LockedRand) *Uniform {
	return &Uniform{
		Min:       min,
		Max:       max,
		generator: generator,
	}
}

// Next returns a random sample drawn from the distribution.
func (u Uniform) Next() float32 {
	return u.generator.Float32()*(u.Max-u.Min) + u.Min
}
