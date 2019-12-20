// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uniform

import "golang.org/x/exp/rand"

// Uniform represents a continuous uniform distribution (https://en.wikipedia.org/wiki/Uniform_distribution_%28continuous%29).
type Uniform struct {
	Min float64
	Max float64
	rnd *rand.Rand
}

func New(min, max float64, source rand.Source) *Uniform {
	u := &Uniform{
		Min: min,
		Max: max,
		rnd: nil,
	}
	if source == nil {
		u.rnd = rand.New(rand.NewSource(1))
	} else {
		u.rnd = rand.New(source)
	}
	return u
}

// Next returns a random sample drawn from the distribution.
func (u Uniform) Next() float64 {
	return u.rnd.Float64()*(u.Max-u.Min) + u.Min
}
