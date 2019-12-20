// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rnd

import (
	"brillion.io/spago/pkg/mat"
	"brillion.io/spago/pkg/mat/rnd/uniform"
	"golang.org/x/exp/rand"
	"math"
)

func Bernoulli(r, c int, prob float64, source rand.Source) mat.Matrix {
	out := mat.NewEmptyDense(r, c)
	dist := uniform.New(0.0, 1.0, source)
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
