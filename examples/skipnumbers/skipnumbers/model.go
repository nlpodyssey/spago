// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package skipnumbers

import (
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var _ nn.Model = &Model{}

// Model that makes predictions using the last state of the recurrent network.
type Model struct {
	RNN       nn.Model
	Predictor nn.Model
}

func NewModel(rnn, predictor nn.Model) *Model {
	return &Model{
		RNN:       rnn,
		Predictor: predictor,
	}
}

func (m *Model) Init() {
	rndGen := rand.NewLockedRand(42)
	nn.ForEachParam(m.RNN, func(param *nn.Param) {
		if param.Type() == nn.Weights {
			// TODO: how to know the right gain for each param? Should the gain be a property of the param itself?
			initializers.XavierUniform(param.Value(), 1, rndGen)
		}
	})
	nn.ForEachParam(m.Predictor, func(param *nn.Param) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), initializers.Gain(ag.OpSoftmax), rndGen)
		}
	})
}

type Processor struct {
	nn.BaseProcessor
	RNN       nn.Processor
	Predictor nn.Processor
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		RNN:       m.RNN.NewProc(ctx),
		Predictor: m.Predictor.NewProc(ctx),
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	return p.Predictor.Forward(p.RNN.Forward(xs...)[len(xs)-1])
}
