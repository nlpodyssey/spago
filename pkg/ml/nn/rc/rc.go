// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rc contains built-in Residual Connections (RC).
package rc

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// PreNorm performs pre-norm residual connections:
//     y = x + F(Norm(x))
func PreNorm[T mat.DType](
	g *ag.Graph[T],
	f func(...ag.Node[T]) []ag.Node[T],
	norm func(...ag.Node[T]) []ag.Node[T],
	xs ...ag.Node[T],
) []ag.Node[T] {
	return add(g, xs, norm(f(xs...)...))
}

// PostNorm performs post-norm residual connections:
//    y = Norm(x + F(x))
func PostNorm[T mat.DType](
	g *ag.Graph[T],
	f func(...ag.Node[T]) []ag.Node[T],
	norm func(...ag.Node[T]) []ag.Node[T],
	xs ...ag.Node[T],
) []ag.Node[T] {
	return norm(add(g, xs, f(xs...))...)
}

// ReZero performs residual connections by rescaling the function with an alpha
// learnable parameter (Bachlechner et al., 2020):
//     y = x + alpha * F(x)
func ReZero[T mat.DType](
	g *ag.Graph[T],
	f func(...ag.Node[T]) []ag.Node[T],
	alpha ag.Node[T],
	xs ...ag.Node[T],
) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, fx := range f(xs...) {
		ys[i] = g.Add(xs[i], g.ProdScalar(fx, alpha))
	}
	return ys
}

func add[T mat.DType](g *ag.Graph[T], a, b []ag.Node[T]) []ag.Node[T] {
	c := make([]ag.Node[T], len(a))
	for i := 0; i < len(a); i++ {
		c[i] = g.Add(a[i], b[i])
	}
	return c
}
