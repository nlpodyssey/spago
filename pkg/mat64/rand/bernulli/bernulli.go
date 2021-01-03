// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bernulli

import (
	"github.com/nlpodyssey/spago/pkg/mat64"
	"github.com/nlpodyssey/spago/pkg/mat64/rand"
	"github.com/nlpodyssey/spago/pkg/mat64/rand/uniform"
	"math"
)

// Distribution creates a new matrix initialized with Bernoulli distribution.
func Distribution(r, c int, prob float64, generator *rand.LockedRand) mat64.Matrix {
	out := mat64.NewEmptyDense(r, c)
	dist := uniform.New(0.0, 1.0, generator)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := dist.Next()
			if val < prob {
				out.Set(i, j, math.Floor(val))
			} else {
				out.Set(i, j, math.Floor(val)+1.0)
			}
		}
	}
	return out
}
