// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Implementation of the Broad Learning System (BLS) described in "Broad Learning System: An Effective and Efficient
Incremental Learning System Without the Need for Deep Architecture" by C. L. Philip Chen and Zhulin Liu, 2017.
(https://ieeexplore.ieee.org/document/7987745)

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
	"io"
	"log"
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

type Model struct {
	Config
	Wz []*nn.Param `type:"weights"`
	Bz []*nn.Param `type:"biases"`
	Wh *nn.Param   `type:"weights"`
	Bh *nn.Param   `type:"biases"`
	W  *nn.Param   `type:"weights"`
	B  *nn.Param   `type:"biases"`
}

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

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

type Processor struct {
	opt   []interface{}
	model *Model
	mode  nn.ProcessingMode
	g     *ag.Graph
	wz    []ag.Node
	bz    []ag.Node
	wh    ag.Node
	bh    ag.Node
	w     ag.Node
	b     ag.Node
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
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
	p := &Processor{
		model: m,
		mode:  nn.Training,
		opt:   opt,
		g:     g,
		wz:    wx,
		bz:    bx,
		wh:    wh,
		bh:    bh,
		w:     g.NewWrap(m.W),
		b:     g.NewWrap(m.B),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("bls: invalid init options")
	}
}

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return false }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.forward(x)
	}
	return ys
}

func (p *Processor) forward(x ag.Node) ag.Node {
	z := p.useFeaturesDropout(p.featuresMapping(x))
	h := p.useEnhancedNodesDropout(p.g.Invoke(p.model.EnhancedNodesActivation, nn.Affine(p.g, p.bh, p.wh, z)))
	y := p.g.Invoke(p.model.OutputActivation, nn.Affine(p.g, p.b, p.w, p.g.Concat([]ag.Node{z, h}...)))
	return y
}

func (p *Processor) featuresMapping(x ag.Node) ag.Node {
	z := make([]ag.Node, p.model.NumOfFeatures)
	for i := range z {
		z[i] = nn.Affine(p.g, p.bz[i], p.wz[i], x)
	}
	return p.g.Invoke(p.model.FeaturesActivation, p.g.Concat(z...))
}

func (p *Processor) useFeaturesDropout(x ag.Node) ag.Node {
	if p.mode == nn.Training && p.model.FeaturesDropout > 0.0 {
		return p.g.Dropout(x, p.model.FeaturesDropout)
	} else {
		return x
	}
}

func (p *Processor) useEnhancedNodesDropout(x ag.Node) ag.Node {
	if p.mode == nn.Training && p.model.EnhancedNodesDropout > 0.0 {
		return p.g.Dropout(x, p.model.EnhancedNodesDropout)
	} else {
		return x
	}
}
