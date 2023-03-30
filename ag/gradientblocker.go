// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Node = &GradientBlocker{}

// GradientBlocker embeds any Node implementation disabling gradients handling and
// blocking gradients accumulation.
type GradientBlocker struct {
	Node
}

// StopGrad creates a new GradientBlocker Node that stops the accumulated gradients from
// flowing through the wrapped Node.
func StopGrad(node Node) Node {
	return &GradientBlocker{
		Node: node,
	}
}

// Grad always returns nil on a GradientBlocker Node.
func (r *GradientBlocker) Grad() mat.Matrix { return nil }

// AccGrad has no effects on a GradientBlocker Node.
func (r *GradientBlocker) AccGrad(mat.Matrix) {}

// HasGrad always returns false on a GradientBlocker Node.
func (r *GradientBlocker) HasGrad() bool { return false }

// RequiresGrad always returns false on a GradientBlocker Node.
func (r *GradientBlocker) RequiresGrad() bool { return false }

// ZeroGrad has no effects on a GradientBlocker Node.
func (r *GradientBlocker) ZeroGrad() {}
