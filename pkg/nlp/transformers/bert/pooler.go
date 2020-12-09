// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model     = &Pooler{}
	_ nn.Processor = &PoolerProcessor{}
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
		Model: stack.New(
			linear.New(config.InputSize, config.OutputSize),
			activation.New(ag.OpTanh),
		),
	}
}

type PoolerProcessor struct {
	*stack.Processor
}

func (m *Pooler) NewProc(ctx nn.Context) nn.Processor {
	return &PoolerProcessor{
		Processor: m.Model.NewProc(ctx).(*stack.Processor),
	}
}
