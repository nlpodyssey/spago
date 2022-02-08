// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bernulli

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/mat/rand/uniform"
)

// Distribution creates a new matrix initialized with Bernoulli distribution.
func Distribution[T mat.DType](r, c int, prob T, generator *rand.LockedRand[T]) mat.Matrix[T] {
	out := mat.NewEmptyDense[T](r, c)
	dist := uniform.New(0.0, 1.0, generator)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := dist.Next()
			if val < prob {
				out.Set(i, j, mat.Floor(val))
			} else {
				out.Set(i, j, mat.Floor(val)+1)
			}
		}
	}
	return out
}
