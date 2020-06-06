// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package transformer

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

type PoolerConfig struct {
	InputSize  int
	OutputSize int
}

type Pooler struct {
	*stack.Model
}

func NewPooler(config PoolerConfig) *Pooler {
	return &Pooler{
		stack.New(
			linear.New(config.InputSize, config.OutputSize),
			activation.New(ag.OpTanh),
		),
	}
}
