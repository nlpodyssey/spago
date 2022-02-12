// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package normal

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
)

// Normal is a source of normally distributed random numbers.
type Normal[T mat.DType] struct {
	Std       T
	Mean      T
	generator *rand.LockedRand[T]
}

// New returns a new Normal, initialized with the given standard deviation and
// mean parameters.
func New[T mat.DType](std, mean T, generator *rand.LockedRand[T]) *Normal[T] {
	return &Normal[T]{
		Std:       std,
		Mean:      mean,
		generator: generator,
	}
}

// Next returns a random sample drawn from the distribution.
func (n Normal[T]) Next() T {
	return n.generator.NormFloat()*n.Std + n.Mean
}
