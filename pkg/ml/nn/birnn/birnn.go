// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"sync"
)

type MergeType int

const (
	Concat MergeType = iota // The outputs are concatenated together (the default)
	Sum                     // The outputs are added together
	Prod                    // The outputs are multiplied element-wise together
	Avg                     // The average of the outputs is taken
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Positive  nn.Model // positive time direction a.k.a. left-to-right
	Negative  nn.Model // negative time direction a.k.a. right-to-left
	MergeMode MergeType
}

// New returns a new model with parameters initialized to zeros.
func New(positive, negative nn.Model, merge MergeType) *Model {
	return &Model{
		Positive:  positive,
		Negative:  negative,
		MergeMode: merge,
	}
}

type Processor struct {
	nn.BaseProcessor
	MergeMode MergeType
	Positive  nn.Processor
	Negative  nn.Processor
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		MergeMode: m.MergeMode,
		Positive:  m.Positive.NewProc(ctx),
		Negative:  m.Negative.NewProc(ctx),
	}
}

// Forward performs the forward step for each input and returns the result.
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
	g := p.Graph
	switch p.MergeMode {
	case Concat:
		return g.Concat(a, b)
	case Sum:
		return g.Add(a, b)
	case Prod:
		return g.Prod(a, b)
	case Avg:
		return g.ProdScalar(g.Add(a, b), g.NewScalar(0.5))
	default:
		panic("birnn: invalid merge mode")
	}
}
