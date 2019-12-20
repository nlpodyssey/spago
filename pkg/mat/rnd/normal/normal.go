// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package normal

import "golang.org/x/exp/rand"

type Normal struct {
	Std  float64
	Mean float64
	rnd  *rand.Rand
}

func New(std, mean float64, source rand.Source) *Normal {
	n := &Normal{
		Std:  std,
		Mean: mean,
		rnd:  nil,
	}
	if source == nil {
		n.rnd = rand.New(rand.NewSource(1))
	} else {
		n.rnd = rand.New(source)
	}
	return n
}

// Next returns a random sample drawn from the distribution.
func (u Normal) Next() float64 {
	return u.rnd.Float64()*u.Std + u.Mean
}
