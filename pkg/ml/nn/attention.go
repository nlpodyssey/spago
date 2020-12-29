// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import "github.com/nlpodyssey/spago/pkg/ml/ag"

// AttentionInput is a set of values suitable as input for an attention function, as described in "Attention Is
// All You Need" (Vaswani et al., 2017 - http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).
type AttentionInput struct {
	Queries []ag.Node
	Keys    []ag.Node
	Values  []ag.Node
}
