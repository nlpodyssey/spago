// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

// PositiveELU returns a new operator node as a result of ELU(x) + 1.
func (g *Graph) PositiveELU(x Node) Node {
	return g.AddScalar(g.ELU(x, g.Constant(1.0)), g.Constant(1.0))
}
