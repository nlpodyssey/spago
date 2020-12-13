// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rc contains built-in Residual Connections (RC).
package rc

import "github.com/nlpodyssey/spago/pkg/ml/ag"

// PreNorm performs pre-norm residual connections:
//     y = x + F(Norm(x))
func PreNorm(
	g *ag.Graph,
	f func(...ag.Node) []ag.Node,
	norm func(...ag.Node) []ag.Node,
	xs ...ag.Node,
) []ag.Node {
	return add(g, xs, norm(f(xs...)...))
}

// PostNorm performs post-norm residual connections:
//    y = Norm(x + F(x))
func PostNorm(
	g *ag.Graph,
	f func(...ag.Node) []ag.Node,
	norm func(...ag.Node) []ag.Node,
	xs ...ag.Node,
) []ag.Node {
	return norm(add(g, xs, f(xs...))...)
}

// ReZero performs residual connections by rescaling the function with an alpha
// learnable parameter (Bachlechner et al., 2020):
//     y = x + alpha * F(x)
func ReZero(
	g *ag.Graph,
	f func(...ag.Node) []ag.Node,
	alpha ag.Node,
	xs ...ag.Node,
) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, fx := range f(xs...) {
		ys[i] = g.Add(xs[i], g.ProdScalar(fx, alpha))
	}
	return ys
}

func add(g *ag.Graph, a, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = g.Add(a[i], b[i])
	}
	return c
}
