// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package normal

import (
	"github.com/nlpodyssey/spago/pkg/mat/rand"
)

type Normal struct {
	Std       float64
	Mean      float64
	generator *rand.LockedRand
}

func New(std, mean float64, generator *rand.LockedRand) *Normal {
	return &Normal{
		Std:       std,
		Mean:      mean,
		generator: generator,
	}
}

// Next returns a random sample drawn from the distribution.
func (u Normal) Next() float64 {
	return u.generator.NormFloat64()*u.Std + u.Mean
}
