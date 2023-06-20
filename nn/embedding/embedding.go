// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedding

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

// Embedding is an implementation of *nn.Param representing embedding values.
type Embedding struct {
	*nn.Param
	m   *Model
	idx int
}

func (e *Embedding) AccGrad(gx mat.Tensor) {
	e.m.mu.Lock()
	defer e.m.mu.Unlock()
	e.m.embedGradIdx[e.idx] = struct{}{}
	e.Param.AccGrad(gx)
}

func (e *Embedding) ZeroGrad() {
	e.m.mu.Lock()
	defer e.m.mu.Unlock()
	delete(e.m.embedGradIdx, e.idx)
	e.Param.ZeroGrad()
}
