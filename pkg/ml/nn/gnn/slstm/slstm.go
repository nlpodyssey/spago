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
	Config                 Config
	InputGate              *HyperLinear4 `type:"params"`
	LeftCellGate           *HyperLinear4 `type:"params"`
	RightCellGate          *HyperLinear4 `type:"params"`
	CellGate               *HyperLinear4 `type:"params"`
	SentCellGate           *HyperLinear4 `type:"params"`
	OutputGate             *HyperLinear4 `type:"params"`
	InputActivation        *HyperLinear4 `type:"params"`
	NonLocalSentCellGate   *HyperLinear3 `type:"params"`
	NonLocalInputGate      *HyperLinear3 `type:"params"`
	NonLocalSentOutputGate *HyperLinear3 `type:"params"`
	StartH                 *nn.Param     `type:"weights"`
	EndH                   *nn.Param     `type:"weights"`
	InitValue              *nn.Param     `type:"weights"`
}

// HyperLinear4 groups multiple params for an affine transformation.
type HyperLinear4 struct {
	W *nn.Param `type:"weights"`
	U *nn.Param `type:"weights"`
	V *nn.Param `type:"weights"`
	B *nn.Param `type:"biases"`
}

// HyperLinear3 groups multiple params for an affine transformation.
type HyperLinear3 struct {
	W *nn.Param `type:"weights"`
	U *nn.Param `type:"weights"`
	B *nn.Param `type:"biases"`
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
		Config:                 config,
		InputGate:              newGate4(in, out),
		LeftCellGate:           newGate4(in, out),
		RightCellGate:          newGate4(in, out),
		CellGate:               newGate4(in, out),
		SentCellGate:           newGate4(in, out),
		OutputGate:             newGate4(in, out),
		InputActivation:        newGate4(in, out),
		NonLocalSentCellGate:   newGate3(out),
		NonLocalInputGate:      newGate3(out),
		NonLocalSentOutputGate: newGate3(out),

		StartH:    nn.NewParam(mat.NewEmptyVecDense(out)),
		EndH:      nn.NewParam(mat.NewEmptyVecDense(out)),
		InitValue: nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

func newGate4(in, out int) *HyperLinear4 {
	return &HyperLinear4{
		W: nn.NewParam(mat.NewEmptyDense(out, out*windowSize)),
		U: nn.NewParam(mat.NewEmptyDense(out, in)),
		V: nn.NewParam(mat.NewEmptyDense(out, out)),
		B: nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

func newGate3(size int) *HyperLinear3 {
	return &HyperLinear3{
		W: nn.NewParam(mat.NewEmptyDense(size, size)),
		U: nn.NewParam(mat.NewEmptyDense(size, size)),
		B: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

type Processor struct {
	nn.BaseProcessor
	Config Config
	// InputGate
	Wi, Ui, Vi, Bi ag.Node
	// LeftCellGate
	Wl, Ul, Vl, Bl ag.Node
	// RightCellGate
	Wr, Ur, Vr, Br ag.Node
	// CellGate
	Wf, Uf, Vf, Bf ag.Node
	// SentCellGate
	Ws, Us, Vs, Bs ag.Node
	// OutputGate
	Wo, Uo, Vo, Bo ag.Node
	// InputActivation
	Wu, Uu, Vu, Bu ag.Node
	// NonLocalSentCellGate
	NLSentWg, NLSentUg, NLSentBg ag.Node
	// NonLocalInputGate
	NLSentWf, NLSentUf, NLSentBf ag.Node
	// NonLocalSentOutputGate
	NLSentWo, NLSentUo, NLSentBo ag.Node
	// Start/End Hidden
	StartH, EndH ag.Node
	// Values at t0
	InitValue ag.Node
	// Shared among all steps
	xUi, xUl, xUr, xUf, xUs, xUo, xUu []ag.Node
	// Shared among all nodes at the same step
	ViPrevG ag.Node
	VlPrevG ag.Node
	VrPrevG ag.Node
	VfPrevG ag.Node
	VsPrevG ag.Node
	VoPrevG ag.Node
	VuPrevG ag.Node
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		Config: m.Config,

		Wi: g.NewWrap(m.InputGate.W),
		Ui: g.NewWrap(m.InputGate.U),
		Vi: g.NewWrap(m.InputGate.V),
		Bi: g.NewWrap(m.InputGate.B),

		Wl: g.NewWrap(m.LeftCellGate.W),
		Ul: g.NewWrap(m.LeftCellGate.U),
		Vl: g.NewWrap(m.LeftCellGate.V),
		Bl: g.NewWrap(m.LeftCellGate.B),

		Wr: g.NewWrap(m.RightCellGate.W),
		Ur: g.NewWrap(m.RightCellGate.U),
		Vr: g.NewWrap(m.RightCellGate.V),
		Br: g.NewWrap(m.RightCellGate.B),

		Wf: g.NewWrap(m.CellGate.W),
		Uf: g.NewWrap(m.CellGate.U),
		Vf: g.NewWrap(m.CellGate.V),
		Bf: g.NewWrap(m.CellGate.B),

		Ws: g.NewWrap(m.SentCellGate.W),
		Us: g.NewWrap(m.SentCellGate.U),
		Vs: g.NewWrap(m.SentCellGate.V),
		Bs: g.NewWrap(m.SentCellGate.B),

		Wo: g.NewWrap(m.OutputGate.W),
		Uo: g.NewWrap(m.OutputGate.U),
		Vo: g.NewWrap(m.OutputGate.V),
		Bo: g.NewWrap(m.OutputGate.B),

		Wu: g.NewWrap(m.InputActivation.W),
		Uu: g.NewWrap(m.InputActivation.U),
		Vu: g.NewWrap(m.InputActivation.V),
		Bu: g.NewWrap(m.InputActivation.B),

		NLSentWg: g.NewWrap(m.NonLocalSentCellGate.W),
		NLSentUg: g.NewWrap(m.NonLocalSentCellGate.U),
		NLSentBg: g.NewWrap(m.NonLocalSentCellGate.B),

		NLSentWf: g.NewWrap(m.NonLocalInputGate.W),
		NLSentUf: g.NewWrap(m.NonLocalInputGate.U),
		NLSentBf: g.NewWrap(m.NonLocalInputGate.B),

		NLSentWo: g.NewWrap(m.NonLocalSentOutputGate.W),
		NLSentUo: g.NewWrap(m.NonLocalSentOutputGate.U),
		NLSentBo: g.NewWrap(m.NonLocalSentOutputGate.B),

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
	fG := g.Sigmoid(nn.Affine(g, p.NLSentBg, p.NLSentWg, prevG, p.NLSentUg, avgH))
	oG := g.Sigmoid(nn.Affine(g, p.NLSentBo, p.NLSentWo, prevG, p.NLSentUo, avgH))

	hG := make([]ag.Node, n)
	gG := nn.Affine(g, p.NLSentBf, p.NLSentWf, prevG)
	for i := 0; i < n; i++ {
		hG[i] = g.Sigmoid(g.Add(gG, g.Mul(p.NLSentUf, prevH[i])))
	}

	var sum ag.Node
	for i := 0; i < n; i++ {
		sum = g.Add(sum, g.Prod(hG[i], prevC[i]))
	}

	cg := g.Add(g.Prod(fG, prevG), sum)
	gt := g.Prod(oG, g.Tanh(cg))
	return gt, cg
}
