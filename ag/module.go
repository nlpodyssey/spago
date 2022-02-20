// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/mat"

// Differentiable must be implemented by all structures requiring autograd capabilities.
type Differentiable[T mat.DType] interface {
	mustEmbedDifferentiableModule()
}

// DifferentiableModule must be embedded into all differentiable modules.
type DifferentiableModule[T mat.DType] struct {
	Graph *Graph[T]
}

func (m DifferentiableModule[T]) mustEmbedDifferentiableModule() {}
