// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uniform

import (
	"github.com/nlpodyssey/spago/pkg/mat/rand"
)

// Uniform represents a continuous uniform distribution (https://en.wikipedia.org/wiki/Uniform_distribution_%28continuous%29).
type Uniform struct {
	Min       float64
	Max       float64
	generator *rand.LockedRand
}

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
