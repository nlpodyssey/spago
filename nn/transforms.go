// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// BiLinear performs a bilinear transformation of the type (x_1 W x_2)
func BiLinear[T mat.DType](w, x1, x2 ag.Node[T]) ag.Node[T] {
	return ag.Mul(ag.Mul(ag.T(x1), w), x2)
}

// BiAffine performs a biaffine transformation.
func BiAffine[T mat.DType](w, u, v, b, x1, x2 ag.Node[T]) ag.Node[T] {
	return ag.Add(ag.Add(ag.Add(BiLinear(w, x1, x2), ag.Mul(ag.T(u), x1)), ag.Mul(ag.T(v), x2)), b)
}
