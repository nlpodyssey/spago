// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package normal

import (
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
)

// Normal is a source of normally distributed random numbers.
type Normal struct {
	Std       float32
	Mean      float32
	generator *rand.LockedRand
}

// New returns a new Normal, initialized with the given standard deviation and
// mean parameters.
func New(std, mean float32, generator *rand.LockedRand) *Normal {
	return &Normal{
		Std:       std,
		Mean:      mean,
		generator: generator,
	}
}

// Next returns a random sample drawn from the distribution.
func (u Normal) Next() float32 {
	return u.generator.NormFloat32()*u.Std + u.Mean
}
