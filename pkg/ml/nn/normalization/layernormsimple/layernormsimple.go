// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernormsimple

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Reference: "Understanding and Improving Layer Normalization" by Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, Junyang Lin (2019).
// (https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf)
type Model struct{}

func New() *Model {
	return &Model{}
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

type Processor struct {
	opt   []interface{}
	model *Model
	mode  nn.ProcessingMode
	g     *ag.Graph
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model: m,
		mode:  nn.Training,
		opt:   opt,
		g:     g,
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("layernormsimple: invalid init options")
	}
}

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return false }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	eps := p.g.NewScalar(1e-10)
	for i, x := range xs {
		mean := p.g.ReduceMean(x)
		stdDev := p.stdDev(x, mean)
		ys[i] = p.g.DivScalar(p.g.SubScalar(x, mean), p.g.Add(stdDev, eps))
	}
	return ys
}

func (p *Processor) stdDev(x ag.Node, mean ag.Node) ag.Node {
	diffVector := p.g.Square(p.g.SubScalar(x, mean))
	return p.g.Sqrt(p.g.ReduceMean(diffVector))
}
