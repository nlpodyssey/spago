// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"io"
	"log"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/nn"
	"sync"
)

type MergeType int

const (
	Concat MergeType = iota // The outputs are concatenated together (the default)
	Sum                     // The outputs are added together
	Prod                    // The outputs are multiplied element-wise together
	Avg                     // The average of the outputs is taken
)

type Model struct {
	Positive  nn.Model // positive time direction a.k.a. left-to-right
	Negative  nn.Model // negative time direction a.k.a. right-to-left
	MergeMode MergeType
}

func New(positive, negative nn.Model, merge MergeType) *Model {
	return &Model{
		Positive:  positive,
		Negative:  negative,
		MergeMode: merge,
	}
}

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

type Processor struct {
	opt      []interface{}
	model    *Model
	g        *ag.Graph
	Positive nn.Processor
	Negative nn.Processor
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:    m,
		Positive: m.Positive.NewProc(g),
		Negative: m.Negative.NewProc(g),
		g:        g,
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("birnn: invalid init options")
	}
}

func (p *Processor) Model() nn.Model {
	return p.model
}

func (p *Processor) Graph() *ag.Graph {
	return p.g
}

func (p *Processor) RequiresFullSeq() bool {
	return true
}

func (p *Processor) Reset() {
	p.Positive.Reset()
	p.Positive.Reset()
	p.init(p.opt)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	var pos []ag.Node
	var neg []ag.Node
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		pos = p.Positive.Forward(xs...)
	}()
	go func() {
		defer wg.Done()
		neg = p.Negative.Forward(reversed(xs)...)
	}()
	wg.Wait()
	out := make([]ag.Node, len(pos))
	for i := 0; i < len(xs); i++ {
		out[i] = p.merge(pos[i], neg[len(out)-1-i])
	}
	return out
}

func reversed(ns []ag.Node) []ag.Node {
	r := make([]ag.Node, len(ns))
	copy(r, ns)
	for i := 0; i < len(r)/2; i++ {
		j := len(r) - i - 1
		r[i], r[j] = r[j], r[i]
	}
	return r
}

func (p *Processor) merge(a, b ag.Node) ag.Node {
	if p.model.MergeMode == Concat {
		return p.g.Concat(a, b)
	} else if p.model.MergeMode == Sum {
		return p.g.Add(a, b)
	} else if p.model.MergeMode == Prod {
		return p.g.Prod(a, b)
	} else if p.model.MergeMode == Avg {
		return p.g.ProdScalar(p.g.Add(a, b), p.g.NewScalar(0.5))
	} else {
		panic("birnn: invalid merge mode")
	}
}
