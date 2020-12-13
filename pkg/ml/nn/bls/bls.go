// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package bls provides an implementation of the Broad Learning System (BLS) described in "Broad Learning System:
An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture" by C. L. Philip Chen
and Zhulin Liu, 2017. (https://ieeexplore.ieee.org/document/7987745)

The "Model" contains only the inference part of the Broad Learning System.
The ridge regression approximation training is performed by the "BroadLearningAlgorithm".

Since the forward pass is built using the computational graph ("ag" a.k.a. auto-grad package), you can train the BLS
through the gradient-descent learning method, like other neural models, updating all the parameters and not just the
output weights as in the original implementation. Cool, isn't it? ;)
*/
package bls

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Config struct {
	InputSize                    int
	FeaturesSize                 int
	NumOfFeatures                int
	EnhancedNodesSize            int
	OutputSize                   int
	FeaturesActivation           ag.OpName
	EnhancedNodesActivation      ag.OpName
	OutputActivation             ag.OpName
	KeepFeaturesParamsFixed      bool
	KeepEnhancedNodesParamsFixed bool
	FeaturesDropout              float64
	EnhancedNodesDropout         float64
}

// Model contains the serializable parameters.
type Model struct {
	Config
	Wz []*nn.Param `type:"weights"`
	Bz []*nn.Param `type:"biases"`
	Wh *nn.Param   `type:"weights"`
	Bh *nn.Param   `type:"biases"`
	W  *nn.Param   `type:"weights"`
	B  *nn.Param   `type:"biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(c Config) *Model {
	length := c.NumOfFeatures
	wz := make([]*nn.Param, length)
	bz := make([]*nn.Param, length)
	for i := 0; i < length; i++ {
		wz[i] = nn.NewParam(mat.NewEmptyDense(c.FeaturesSize, c.InputSize))
		bz[i] = nn.NewParam(mat.NewEmptyVecDense(c.FeaturesSize))
	}
	return &Model{
		Config: c,
		Wz:     wz,
		Bz:     bz,
		Wh:     nn.NewParam(mat.NewEmptyDense(c.EnhancedNodesSize, c.NumOfFeatures*c.FeaturesSize)),
		Bh:     nn.NewParam(mat.NewEmptyVecDense(c.EnhancedNodesSize)),
		W:      nn.NewParam(mat.NewEmptyDense(c.OutputSize, c.NumOfFeatures*c.FeaturesSize+c.EnhancedNodesSize)),
		B:      nn.NewParam(mat.NewEmptyVecDense(c.OutputSize)),
	}
}

type Processor struct {
	nn.BaseProcessor
	Config
	wz []ag.Node
	bz []ag.Node
	wh ag.Node
	bh ag.Node
	w  ag.Node
	b  ag.Node
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	length := m.NumOfFeatures
	wx := make([]ag.Node, length)
	bx := make([]ag.Node, length)
	if m.KeepFeaturesParamsFixed {
		for i := 0; i < length; i++ {
			wx[i] = g.NewWrapNoGrad(m.Wz[i])
			bx[i] = g.NewWrapNoGrad(m.Bz[i])
		}
	} else {
		for i := 0; i < length; i++ {
			wx[i] = g.NewWrap(m.Wz[i])
			bx[i] = g.NewWrap(m.Bz[i])
		}
	}
	var wh, bh ag.Node
	if m.KeepEnhancedNodesParamsFixed {
		wh = g.NewWrapNoGrad(m.Wh)
		bh = g.NewWrapNoGrad(m.Bh)
	} else {
		wh = g.NewWrap(m.Wh)
		bh = g.NewWrap(m.Bh)
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		Config: m.Config,
		wz:     wx,
		bz:     bx,
		wh:     wh,
		bh:     bh,
		w:      g.NewWrap(m.W),
		b:      g.NewWrap(m.B),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.forward(x)
	}
	return ys
}

func (p *Processor) forward(x ag.Node) ag.Node {
	g := p.Graph
	z := p.useFeaturesDropout(p.featuresMapping(x))
	h := p.useEnhancedNodesDropout(g.Invoke(p.EnhancedNodesActivation, nn.Affine(g, p.bh, p.wh, z)))
	y := g.Invoke(p.OutputActivation, nn.Affine(g, p.b, p.w, g.Concat([]ag.Node{z, h}...)))
	return y
}

func (p *Processor) featuresMapping(x ag.Node) ag.Node {
	z := make([]ag.Node, p.NumOfFeatures)
	for i := range z {
		z[i] = nn.Affine(p.Graph, p.bz[i], p.wz[i], x)
	}
	return p.Graph.Invoke(p.FeaturesActivation, p.Graph.Concat(z...))
}

func (p *Processor) useFeaturesDropout(x ag.Node) ag.Node {
	if p.Mode == nn.Training && p.FeaturesDropout > 0.0 {
		return p.Graph.Dropout(x, p.FeaturesDropout)
	}
	return x
}

func (p *Processor) useEnhancedNodesDropout(x ag.Node) ag.Node {
	if p.Mode == nn.Training && p.EnhancedNodesDropout > 0.0 {
		return p.Graph.Dropout(x, p.EnhancedNodesDropout)
	}
	return x
}
