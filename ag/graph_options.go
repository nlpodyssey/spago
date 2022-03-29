// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
)

// GraphOption allows to configure a new Graph with your specific needs.
type GraphOption[T mat.DType] func(*Graph[T])

// WithRand sets the generator of random numbers.
func WithRand[T mat.DType](rand *rand.LockedRand[T]) GraphOption[T] {
	return func(g *Graph[T]) {
		g.randGen = rand
	}
}

// WithRandSeed set a new generator of random numbers with the given seed.
func WithRandSeed[T mat.DType](seed uint64) GraphOption[T] {
	return func(g *Graph[T]) {
		g.randGen = rand.NewLockedRand[T](seed)
	}
}
