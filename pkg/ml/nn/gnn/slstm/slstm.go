// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// slstm
// Reference: "Sentence-State LSTM for Text Representation" by Zhang et al, 2018.
// (https://arxiv.org/pdf/1805.02474.pdf)
package slstm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// TODO(1): code refactoring using a structure to maintain states.
// TODO(2): use a gradient policy (i.e. reinforcement learning) to increase the context with dynamic skip connections.

// Model contains the serializable parameters.
type Model struct {
	Config Config

	// Input gate
	Wi *nn.Param `type:"weights"`
	Ui *nn.Param `type:"weights"`
	Vi *nn.Param `type:"weights"`
	Bi *nn.Param `type:"biases"`

	// Left context cell gate
	Wl *nn.Param `type:"weights"`
	Ul *nn.Param `type:"weights"`
	Vl *nn.Param `type:"weights"`
	Bl *nn.Param `type:"biases"`

	// Right context cell gate
	Wr *nn.Param `type:"weights"`
	Ur *nn.Param `type:"weights"`
	Vr *nn.Param `type:"weights"`
	Br *nn.Param `type:"biases"`

	// Cell gate
	Wf *nn.Param `type:"weights"`
	Uf *nn.Param `type:"weights"`
	Vf *nn.Param `type:"weights"`
	Bf *nn.Param `type:"biases"`

	// Sentence context cell gate
	Ws *nn.Param `type:"weights"`
	Us *nn.Param `type:"weights"`
	Vs *nn.Param `type:"weights"`
	Bs *nn.Param `type:"biases"`

	// Output gate
	Wo *nn.Param `type:"weights"`
	Uo *nn.Param `type:"weights"`
	Vo *nn.Param `type:"weights"`
	Bo *nn.Param `type:"biases"`

	// Tanh gate
	Wu *nn.Param `type:"weights"`
	Uu *nn.Param `type:"weights"`
	Vu *nn.Param `type:"weights"`
	Bu *nn.Param `type:"biases"`

	SentWg *nn.Param `type:"weights"`
	SentUg *nn.Param `type:"weights"`
	SentBg *nn.Param `type:"biases"`

	SentWf *nn.Param `type:"weights"`
	SentUf *nn.Param `type:"weights"`
	SentBf *nn.Param `type:"biases"`

	SentWo *nn.Param `type:"weights"`
	SentUo *nn.Param `type:"weights"`
	SentBo *nn.Param `type:"biases"`

	StartH    *nn.Param `type:"weights"`
	EndH      *nn.Param `type:"weights"`
	InitValue *nn.Param `type:"weights"`
}

type Config struct {
	InputSize  int
	OutputSize int
	Steps      int
}

const windowSize = 3 // the window is fixed in this implementation

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	in, out := config.InputSize, config.OutputSize
	return &Model{
		Config: config,
		Wi:     nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		Ui:     nn.NewParam(mat.NewEmptyDense(out, in)),
		Vi:     nn.NewParam(mat.NewEmptyDense(out, out)),
		Bi:     nn.NewParam(mat.NewEmptyVecDense(out)),

		Wl: nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		Ul: nn.NewParam(mat.NewEmptyDense(out, in)),
		Vl: nn.NewParam(mat.NewEmptyDense(out, out)),
		Bl: nn.NewParam(mat.NewEmptyVecDense(out)),

		Wr: nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		Ur: nn.NewParam(mat.NewEmptyDense(out, in)),
		Vr: nn.NewParam(mat.NewEmptyDense(out, out)),
		Br: nn.NewParam(mat.NewEmptyVecDense(out)),

		Wf: nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		Uf: nn.NewParam(mat.NewEmptyDense(out, in)),
		Vf: nn.NewParam(mat.NewEmptyDense(out, out)),
		Bf: nn.NewParam(mat.NewEmptyVecDense(out)),

		Ws: nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		Us: nn.NewParam(mat.NewEmptyDense(out, in)),
		Vs: nn.NewParam(mat.NewEmptyDense(out, out)),
		Bs: nn.NewParam(mat.NewEmptyVecDense(out)),

		Wo: nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		Uo: nn.NewParam(mat.NewEmptyDense(out, in)),
		Vo: nn.NewParam(mat.NewEmptyDense(out, out)),
		Bo: nn.NewParam(mat.NewEmptyVecDense(out)),

		Wu: nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		Uu: nn.NewParam(mat.NewEmptyDense(out, in)),
		Vu: nn.NewParam(mat.NewEmptyDense(out, out)),
		Bu: nn.NewParam(mat.NewEmptyVecDense(out)),

		SentWg: nn.NewParam(mat.NewEmptyDense(out, out)),
		SentUg: nn.NewParam(mat.NewEmptyDense(out, out)),
		SentBg: nn.NewParam(mat.NewEmptyVecDense(out)),

		SentWf: nn.NewParam(mat.NewEmptyDense(out, out)),
		SentUf: nn.NewParam(mat.NewEmptyDense(out, out)),
		SentBf: nn.NewParam(mat.NewEmptyVecDense(out)),

		SentWo: nn.NewParam(mat.NewEmptyDense(out, out)),
		SentUo: nn.NewParam(mat.NewEmptyDense(out, out)),
		SentBo: nn.NewParam(mat.NewEmptyVecDense(out)),

		StartH:    nn.NewParam(mat.NewEmptyVecDense(out)),
		EndH:      nn.NewParam(mat.NewEmptyVecDense(out)),
		InitValue: nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

type Processor struct {
	nn.BaseProcessor
	Config                 Config
	Wi, Ui, Vi, Bi         ag.Node
	Wl, Ul, Vl, Bl         ag.Node
	Wr, Ur, Vr, Br         ag.Node
	Wf, Uf, Vf, Bf         ag.Node
	Ws, Us, Vs, Bs         ag.Node
	Wo, Uo, Vo, Bo         ag.Node
	Wu, Uu, Vu, Bu         ag.Node
	SentWg, SentUg, SentBg ag.Node
	SentWf, SentUf, SentBf ag.Node
	SentWo, SentUo, SentBo ag.Node
	StartH, EndH           ag.Node
	InitValue              ag.Node
	// shared among all steps
	xUi, xUl, xUr, xUf, xUs, xUo, xUu []ag.Node
	// shared among all nodes at the same step
	ViPrevG ag.Node
	VlPrevG ag.Node
	VrPrevG ag.Node
	VfPrevG ag.Node
	VsPrevG ag.Node
	VoPrevG ag.Node
	VuPrevG ag.Node
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		Config: m.Config,
		Wi:     g.NewWrap(m.Wi),
		Ui:     g.NewWrap(m.Ui),
		Vi:     g.NewWrap(m.Vi),
		Bi:     g.NewWrap(m.Bi),

		Wl: g.NewWrap(m.Wl),
		Ul: g.NewWrap(m.Ul),
		Vl: g.NewWrap(m.Vl),
		Bl: g.NewWrap(m.Bl),

		Wr: g.NewWrap(m.Wr),
		Ur: g.NewWrap(m.Ur),
		Vr: g.NewWrap(m.Vr),
		Br: g.NewWrap(m.Br),

		Wf: g.NewWrap(m.Wf),
		Uf: g.NewWrap(m.Uf),
		Vf: g.NewWrap(m.Vf),
		Bf: g.NewWrap(m.Bf),

		Ws: g.NewWrap(m.Ws),
		Us: g.NewWrap(m.Us),
		Vs: g.NewWrap(m.Vs),
		Bs: g.NewWrap(m.Bs),

		Wo: g.NewWrap(m.Wo),
		Uo: g.NewWrap(m.Uo),
		Vo: g.NewWrap(m.Vo),
		Bo: g.NewWrap(m.Bo),

		Wu: g.NewWrap(m.Wu),
		Uu: g.NewWrap(m.Uu),
		Vu: g.NewWrap(m.Vu),
		Bu: g.NewWrap(m.Bu),

		SentWg: g.NewWrap(m.SentWg),
		SentUg: g.NewWrap(m.SentUg),
		SentBg: g.NewWrap(m.SentBg),

		SentWf: g.NewWrap(m.SentWf),
		SentUf: g.NewWrap(m.SentUf),
		SentBf: g.NewWrap(m.SentBf),

		SentWo: g.NewWrap(m.SentWo),
		SentUo: g.NewWrap(m.SentUo),
		SentBo: g.NewWrap(m.SentBo),

		StartH:    g.NewWrap(m.StartH),
		EndH:      g.NewWrap(m.EndH),
		InitValue: g.NewWrap(m.InitValue),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	steps := p.Config.Steps
	n := len(xs)
	h := make([][]ag.Node, steps)
	c := make([][]ag.Node, steps)
	g := make([]ag.Node, steps)
	cg := make([]ag.Node, steps)
	h[0] = make([]ag.Node, n)
	c[0] = make([]ag.Node, n)

	g[0] = p.InitValue
	cg[0] = p.InitValue
	for i := 0; i < n; i++ {
		h[0][i] = p.InitValue
		c[0][i] = p.InitValue
	}

	p.computeUx(xs) // the result is shared among all steps
	for t := 1; t < p.Config.Steps; t++ {
		p.computeVg(g[t-1]) // the result is shared among all nodes
		h[t], c[t] = p.updateHiddenNodes(h[t-1], c[t-1], g[t-1])
		g[t], cg[t] = p.updateSentenceState(h[t-1], c[t-1], g[t-1])
	}

	return h[len(h)-1]
}

func (p *Processor) computeUx(xs []ag.Node) {
	n := len(xs)
	p.xUi = make([]ag.Node, n)
	p.xUl = make([]ag.Node, n)
	p.xUr = make([]ag.Node, n)
	p.xUf = make([]ag.Node, n)
	p.xUs = make([]ag.Node, n)
	p.xUo = make([]ag.Node, n)
	p.xUu = make([]ag.Node, n)

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			p.xUi[i] = p.Graph.Mul(p.Ui, xs[i])
			p.xUl[i] = p.Graph.Mul(p.Ul, xs[i])
			p.xUr[i] = p.Graph.Mul(p.Ur, xs[i])
			p.xUf[i] = p.Graph.Mul(p.Uf, xs[i])
			p.xUs[i] = p.Graph.Mul(p.Us, xs[i])
			p.xUo[i] = p.Graph.Mul(p.Uo, xs[i])
			p.xUu[i] = p.Graph.Mul(p.Uu, xs[i])
		}(i)
	}
	wg.Wait()
}

func (p *Processor) computeVg(prevG ag.Node) {
	var wg sync.WaitGroup
	wg.Add(7)
	for i := 0; i < 7; i++ {
		go func(i int) {
			defer wg.Done()
			switch i {
			case 0:
				p.ViPrevG = p.Graph.Mul(p.Vi, prevG)
			case 1:
				p.VlPrevG = p.Graph.Mul(p.Vl, prevG)
			case 2:
				p.VrPrevG = p.Graph.Mul(p.Vr, prevG)
			case 3:
				p.VfPrevG = p.Graph.Mul(p.Vf, prevG)
			case 4:
				p.VsPrevG = p.Graph.Mul(p.Vs, prevG)
			case 5:
				p.VoPrevG = p.Graph.Mul(p.Vo, prevG)
			case 6:
				p.VuPrevG = p.Graph.Mul(p.Vu, prevG)
			}
		}(i)
	}
	wg.Wait()
}

func (p *Processor) processNode(i int, prevH []ag.Node, prevC []ag.Node, prevG ag.Node) (h ag.Node, c ag.Node) {
	g := p.Graph
	n := len(prevH)
	first := 0
	last := n - 1
	j := i - 1
	k := i + 1

	prevHj, prevCj := func() (ag.Node, ag.Node) {
		if j < first {
			return p.StartH, p.StartH
		}
		return prevH[j], prevC[j]
	}()

	prevHk, prevCk := func() (ag.Node, ag.Node) {
		if k > last {
			return p.EndH, p.EndH
		}
		return prevH[k], prevC[k]
	}()

	context := g.Concat(prevHj, prevH[i], prevHk)
	iG := g.Sigmoid(g.Sum(p.Bi, g.Mul(p.Wi, context), p.ViPrevG, p.xUi[i]))
	lG := g.Sigmoid(g.Sum(p.Bl, g.Mul(p.Wl, context), p.VlPrevG, p.xUl[i]))
	rG := g.Sigmoid(g.Sum(p.Br, g.Mul(p.Wr, context), p.VrPrevG, p.xUr[i]))
	fG := g.Sigmoid(g.Sum(p.Bf, g.Mul(p.Wf, context), p.VfPrevG, p.xUf[i]))
	sG := g.Sigmoid(g.Sum(p.Bs, g.Mul(p.Ws, context), p.VsPrevG, p.xUs[i]))
	oG := g.Sigmoid(g.Sum(p.Bo, g.Mul(p.Wo, context), p.VoPrevG, p.xUo[i]))
	uG := g.Tanh(g.Sum(p.Bu, g.Mul(p.Wu, context), p.VuPrevG, p.xUu[i]))
	c1 := g.Prod(lG, prevCj)
	c2 := g.Prod(fG, prevC[i])
	c3 := g.Prod(rG, prevCk)
	c4 := g.Prod(sG, prevG)
	c5 := g.Prod(iG, uG)
	c = g.Sum(c1, c2, c3, c4, c5)
	h = g.Prod(oG, g.Tanh(c))
	return
}

func (p *Processor) updateHiddenNodes(prevH []ag.Node, prevC []ag.Node, prevG ag.Node) ([]ag.Node, []ag.Node) {
	var wg sync.WaitGroup
	n := len(prevH)
	wg.Add(n)
	h := make([]ag.Node, n)
	c := make([]ag.Node, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			h[i], c[i] = p.processNode(i, prevH, prevC, prevG)
		}(i)
	}
	wg.Wait()
	return h, c
}

func (p *Processor) updateSentenceState(prevH []ag.Node, prevC []ag.Node, prevG ag.Node) (ag.Node, ag.Node) {
	g := p.Graph
	n := len(prevH)
	avgH := g.Mean(prevH)
	fG := g.Sigmoid(nn.Affine(g, p.SentBg, p.SentWg, prevG, p.SentUg, avgH))
	oG := g.Sigmoid(nn.Affine(g, p.SentBo, p.SentWo, prevG, p.SentUo, avgH))

	hG := make([]ag.Node, n)
	gG := nn.Affine(g, p.SentBf, p.SentWf, prevG)
	for i := 0; i < n; i++ {
		hG[i] = g.Sigmoid(g.Add(gG, g.Mul(p.SentUf, prevH[i])))
	}

	var sum ag.Node
	for i := 0; i < n; i++ {
		sum = g.Add(sum, g.Prod(hG[i], prevC[i]))
	}

	cg := g.Add(g.Prod(fG, prevG), sum)
	gt := g.Prod(oG, g.Tanh(cg))
	return gt, cg
}
