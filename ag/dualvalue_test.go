// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
)

type dummyNode struct {
	id           int // just an identifier for testing and debugging
	value        mat.Matrix
	grad         mat.Matrix
	requiresGrad bool
}

func (n *dummyNode) Foo()               { panic("not implemented") }
func (n *dummyNode) Value() mat.Matrix  { return n.value }
func (n *dummyNode) Grad() mat.Matrix   { return n.grad }
func (n *dummyNode) HasGrad() bool      { return n.grad != nil }
func (n *dummyNode) RequiresGrad() bool { return n.requiresGrad }
func (n *dummyNode) AccGrad(mat.Matrix) { panic("not implemented") }
func (n *dummyNode) ZeroGrad()          { panic("not implemented") }
func (n *dummyNode) Name() string       { panic("not implemented") }
