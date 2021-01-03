// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bernulli

import (
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/mat32/rand/uniform"
	"math"
)

// Distribution creates a new matrix initialized with Bernoulli distribution.
func Distribution(r, c int, prob float32, generator *rand.LockedRand) mat32.Matrix {
	out := mat32.NewEmptyDense(r, c)
	dist := uniform.New(0.0, 1.0, generator)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := dist.Next()
			if val < prob {
				out.Set(i, j, float32(math.Floor(float64(val))))
			} else {
				out.Set(i, j, float32(math.Floor(float64(val)))+1.0)
			}
		}
	}
	return out
}
