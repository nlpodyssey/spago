// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bernulli

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/mat/rand/uniform"
	"math"
)

func Distribution(r, c int, prob float64, generator *rand.LockedRand) mat.Matrix {
	out := mat.NewEmptyDense(r, c)
	dist := uniform.New(0.0, 1.0, generator)
	for i := 0; i < out.Size(); i++ {
		val := dist.Next()
		if val < prob {
			out.Set(math.Floor(val), i)
		} else {
			out.Set(math.Floor(val)+1.0, i)
		}
	}
	return out
}
