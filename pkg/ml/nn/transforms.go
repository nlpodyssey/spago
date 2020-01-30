// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"saientist.dev/spago/pkg/ml/ag"
)

// Linear performs a linear transformation of the type Wx.
func Linear(g *ag.Graph, w, x ag.Node) ag.Node {
	return g.Mul(w, x)
}

// Affine performs an affine transformation over an arbitrary (odd) number of nodes held in the input.
// The first node is the “bias”, which is added to the output as-is.
// The remaining nodes of the form "Wx" are multiplied together in pairs, then added.
// The pairs except the first whose "x" is nil are not considered.
// y = b + W1x1 + W2x2 + ... + WnXn
func Affine(g *ag.Graph, xs ...ag.Node) ag.Node {
	if len(xs)%2 == 0 {
		panic("nn: the number of arguments of the affine transformation should be odd")
	}
	y := g.Add(xs[0], Linear(g, xs[1], xs[2])) // b + Wx
	for i := 3; i < len(xs)-1; i += 2 {
		w := xs[i]
		x := xs[i+1]
		if x != nil {
			y = g.Add(y, Linear(g, w, x))
		}
	}
	return y
}

// BiLinear performs a bilinear transformation of the type (x_1 W x_2)
func BiLinear(g *ag.Graph, w, x1, x2 ag.Node) ag.Node {
	return g.Mul(g.Mul(g.T(x1), w), x2)
}

// BiAffine performs a biaffine transformation.
func BiAffine(g *ag.Graph, w, u, v, b, x1, x2 ag.Node) ag.Node {
	return g.Add(g.Add(g.Add(BiLinear(g, w, x1, x2), g.Mul(g.T(u), x1)), g.Mul(g.T(v), x2)), b)
}

// Conv2D performs a 2D convolution.
func Conv2D(g *ag.Graph, w, x ag.Node, xStride, yStride int) ag.Node {
	var dimx, dimy int
	if (x.Value().Rows()-w.Value().Rows())%xStride != 0 {
		panic("Incompatible stride value for rows")
	}
	if (x.Value().Columns()-w.Value().Columns())%yStride != 0 {
		panic("Incompatible stride value for columns")
	}
	dimx = (x.Value().Rows()-w.Value().Rows())/xStride + 1
	dimy = (x.Value().Columns()-w.Value().Columns())/yStride + 1

	var outList []ag.Node
	for i := 0; i < dimx; i++ {
		for j := 0; j < dimy; j++ {
			var view = g.View(x, i*xStride, j*yStride, w.Value().Rows(), w.Value().Columns())
			var dotProduct = g.Dot(view, w)
			outList = append(outList, dotProduct)
		}
	}

	return g.Reshape(g.Concat(outList...), dimx, dimy)
}
