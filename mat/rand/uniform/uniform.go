// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uniform

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
)

// Uniform is a source of uniformly distributed random numbers.
// See: https://en.wikipedia.org/wiki/Continuous_uniform_distribution.
type Uniform[T mat.DType] struct {
	Min       T
	Max       T
	generator *rand.LockedRand
}

// New returns a new Normal, initialized with the given min and max parameters.
func New[T mat.DType](min, max T, generator *rand.LockedRand) *Uniform[T] {
	return &Uniform[T]{
		Min:       min,
		Max:       max,
		generator: generator,
	}
}

// Next returns a random sample drawn from the distribution.
func (u Uniform[T]) Next() T {
	return rand.LockedRandFloat[T](u.generator)*(u.Max-u.Min) + u.Min
}
