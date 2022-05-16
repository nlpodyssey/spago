// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/internal/rand"
)

// Float returns, as a T, a pseudo-random number in [0.0,1.0)
// from the default Source.
func Float[T float.DType]() T {
	switch any(T(0)).(type) {
	case float32:
		return T(rand.Float32())
	case float64:
		return T(rand.Float64())
	default:
		panic(fmt.Sprintf("rand: unexpected type %T", T(0)))
	}
}
