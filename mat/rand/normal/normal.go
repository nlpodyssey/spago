// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package normal

import "github.com/nlpodyssey/spago/mat/rand"

// Normal is a source of normally distributed random numbers.
type Normal struct {
	Std       float64
	Mean      float64
	generator *rand.LockedRand
}

// New returns a new Normal, initialized with the given standard deviation and
// mean parameters.
func New(std, mean float64, generator *rand.LockedRand) *Normal {
	return &Normal{
		Std:       std,
		Mean:      mean,
		generator: generator,
	}
}

// Next returns a random sample drawn from the distribution.
func (n Normal) Next() float64 {
	return n.generator.NormFloat64()*n.Std + n.Mean
}
