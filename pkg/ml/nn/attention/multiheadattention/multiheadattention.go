// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Attention   []*selfattention.Model
	OutputMerge *linear.Model
	NumOfHeads  int // number of heads
	Dm          int // input and output vectors dimension
	Dk          int // hidden vectors dimension (Dm / NumOfHeads)
}

// New returns a new model with parameters initialized to zeros.
func New(size, numOfHeads int, useCausalMask bool) *Model {
	dm := size
	dk := size / numOfHeads
	attention := make([]*selfattention.Model, numOfHeads)
	attentionConfig := selfattention.Config{
		InputSize:     dm,
		QuerySize:     dk,
		KeySize:       dk,
		ValueSize:     dk,
		ScaleFactor:   1.0 / mat.Sqrt(mat.Float(dk)),
		UseCausalMask: useCausalMask,
	}
	for i := 0; i < numOfHeads; i++ {
		attention[i] = selfattention.New(attentionConfig)
	}
	return &Model{
		Attention:   attention,
		OutputMerge: linear.New(dk*numOfHeads, dm),
		NumOfHeads:  numOfHeads,
		Dm:          dm,
		Dk:          dk,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(attIn attention.QKV) []ag.Node {
	g := m.Graph()
	headsAttention := make([][]ag.Node, m.NumOfHeads)
	for h, proc := range m.Attention {
		headsAttention[h] = proc.Forward(attIn)
	}
	concatHeads := make([]ag.Node, len(attIn.Queries))
	for i := 0; i < len(concatHeads); i++ {
		buf := make([]ag.Node, m.NumOfHeads)
		for j := 0; j < m.NumOfHeads; j++ {
			buf[j] = headsAttention[j][i]
		}
		concatHeads[i] = g.Concat(buf...)
	}
	return m.OutputMerge.Forward(concatHeads...)
}
