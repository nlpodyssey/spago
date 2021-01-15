// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package slstm implements a Sentence-State LSTM graph neural network.
//
// Reference: "Sentence-State LSTM for Text Representation" by Zhang et al, 2018.
// (https://arxiv.org/pdf/1805.02474.pdf)
package slstm

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"sync"
)

var (
	_ nn.Model = &Model{}
)

// TODO(1): code refactoring using a structure to maintain states.
// TODO(2): use a gradient policy (i.e. reinforcement learning) to increase the context with dynamic skip connections.

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config                 Config
	InputGate              *HyperLinear4 `spago:"type:params"`
	LeftCellGate           *HyperLinear4 `spago:"type:params"`
	RightCellGate          *HyperLinear4 `spago:"type:params"`
	CellGate               *HyperLinear4 `spago:"type:params"`
	SentCellGate           *HyperLinear4 `spago:"type:params"`
	OutputGate             *HyperLinear4 `spago:"type:params"`
	InputActivation        *HyperLinear4 `spago:"type:params"`
	NonLocalSentCellGate   *HyperLinear3 `spago:"type:params"`
	NonLocalInputGate      *HyperLinear3 `spago:"type:params"`
	NonLocalSentOutputGate *HyperLinear3 `spago:"type:params"`
	StartH                 nn.Param      `spago:"type:weights"`
	EndH                   nn.Param      `spago:"type:weights"`
	InitValue              nn.Param      `spago:"type:weights"`
	Support                Support       `spago:"scope:processor"`
}

// Config provides configuration settings for a Sentence-State LSTM Model.
type Config struct {
	InputSize  int
	OutputSize int
	Steps      int
}

const windowSize = 3 // the window is fixed in this implementation

// HyperLinear4 groups multiple params for an affine transformation.
type HyperLinear4 struct {
	W nn.Param `spago:"type:weights"`
	U nn.Param `spago:"type:weights"`
	V nn.Param `spago:"type:weights"`
	B nn.Param `spago:"type:biases"`
}

// HyperLinear3 groups multiple params for an affine transformation.
type HyperLinear3 struct {
	W nn.Param `spago:"type:weights"`
	U nn.Param `spago:"type:weights"`
	B nn.Param `spago:"type:biases"`
}

// Support contains nodes used during the forward step.
type Support struct {
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

func init() {
	gob.Register(&Model{})
}

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
		StartH:                 nn.NewParam(mat.NewEmptyVecDense(out)),
		EndH:                   nn.NewParam(mat.NewEmptyVecDense(out)),
		InitValue:              nn.NewParam(mat.NewEmptyVecDense(out)),
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

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	steps := m.Config.Steps
	n := len(xs)
	h := make([][]ag.Node, steps)
	c := make([][]ag.Node, steps)
	g := make([]ag.Node, steps)
	cg := make([]ag.Node, steps)
	h[0] = make([]ag.Node, n)
	c[0] = make([]ag.Node, n)

	g[0] = m.InitValue
	cg[0] = m.InitValue
	for i := 0; i < n; i++ {
		h[0][i] = m.InitValue
		c[0][i] = m.InitValue
	}

	m.computeUx(xs) // the result is shared among all steps
	for t := 1; t < m.Config.Steps; t++ {
		m.computeVg(g[t-1]) // the result is shared among all nodes
		h[t], c[t] = m.updateHiddenNodes(h[t-1], c[t-1], g[t-1])
		g[t], cg[t] = m.updateSentenceState(h[t-1], c[t-1], g[t-1])
	}

	return h[len(h)-1]
}

func (m *Model) computeUx(xs []ag.Node) {
	g := m.Graph()
	n := len(xs)
	m.Support.xUi = make([]ag.Node, n)
	m.Support.xUl = make([]ag.Node, n)
	m.Support.xUr = make([]ag.Node, n)
	m.Support.xUf = make([]ag.Node, n)
	m.Support.xUs = make([]ag.Node, n)
	m.Support.xUo = make([]ag.Node, n)
	m.Support.xUu = make([]ag.Node, n)

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			m.Support.xUi[i] = g.Mul(m.InputGate.U, xs[i])
			m.Support.xUl[i] = g.Mul(m.LeftCellGate.U, xs[i])
			m.Support.xUr[i] = g.Mul(m.RightCellGate.U, xs[i])
			m.Support.xUf[i] = g.Mul(m.CellGate.U, xs[i])
			m.Support.xUs[i] = g.Mul(m.SentCellGate.U, xs[i])
			m.Support.xUo[i] = g.Mul(m.OutputGate.U, xs[i])
			m.Support.xUu[i] = g.Mul(m.InputActivation.U, xs[i])
		}(i)
	}
	wg.Wait()
}

func (m *Model) computeVg(prevG ag.Node) {
	g := m.Graph()
	var wg sync.WaitGroup
	wg.Add(7)
	for i := 0; i < 7; i++ {
		go func(i int) {
			defer wg.Done()
			switch i {
			case 0:
				m.Support.ViPrevG = g.Mul(m.InputGate.V, prevG)
			case 1:
				m.Support.VlPrevG = g.Mul(m.LeftCellGate.V, prevG)
			case 2:
				m.Support.VrPrevG = g.Mul(m.RightCellGate.V, prevG)
			case 3:
				m.Support.VfPrevG = g.Mul(m.CellGate.V, prevG)
			case 4:
				m.Support.VsPrevG = g.Mul(m.SentCellGate.V, prevG)
			case 5:
				m.Support.VoPrevG = g.Mul(m.OutputGate.V, prevG)
			case 6:
				m.Support.VuPrevG = g.Mul(m.InputActivation.U, prevG)
			}
		}(i)
	}
	wg.Wait()
}

func (m *Model) processNode(i int, prevH []ag.Node, prevC []ag.Node, prevG ag.Node) (h ag.Node, c ag.Node) {
	g := m.Graph()
	n := len(prevH)
	first := 0
	last := n - 1
	j := i - 1
	k := i + 1

	prevHj, prevCj := func() (ag.Node, ag.Node) {
		if j < first {
			return m.StartH, m.StartH
		}
		return prevH[j], prevC[j]
	}()

	prevHk, prevCk := func() (ag.Node, ag.Node) {
		if k > last {
			return m.EndH, m.EndH
		}
		return prevH[k], prevC[k]
	}()

	context := g.Concat(prevHj, prevH[i], prevHk)
	iG := g.Sigmoid(g.Sum(m.InputGate.B, g.Mul(m.InputGate.W, context), m.Support.ViPrevG, m.Support.xUi[i]))
	lG := g.Sigmoid(g.Sum(m.LeftCellGate.B, g.Mul(m.LeftCellGate.W, context), m.Support.VlPrevG, m.Support.xUl[i]))
	rG := g.Sigmoid(g.Sum(m.RightCellGate.B, g.Mul(m.RightCellGate.W, context), m.Support.VrPrevG, m.Support.xUr[i]))
	fG := g.Sigmoid(g.Sum(m.CellGate.B, g.Mul(m.CellGate.W, context), m.Support.VfPrevG, m.Support.xUf[i]))
	sG := g.Sigmoid(g.Sum(m.SentCellGate.B, g.Mul(m.SentCellGate.W, context), m.Support.VsPrevG, m.Support.xUs[i]))
	oG := g.Sigmoid(g.Sum(m.OutputGate.B, g.Mul(m.OutputGate.W, context), m.Support.VoPrevG, m.Support.xUo[i]))
	uG := g.Tanh(g.Sum(m.InputActivation.B, g.Mul(m.InputActivation.W, context), m.Support.VuPrevG, m.Support.xUu[i]))
	c1 := g.Prod(lG, prevCj)
	c2 := g.Prod(fG, prevC[i])
	c3 := g.Prod(rG, prevCk)
	c4 := g.Prod(sG, prevG)
	c5 := g.Prod(iG, uG)
	c = g.Sum(c1, c2, c3, c4, c5)
	h = g.Prod(oG, g.Tanh(c))
	return
}

func (m *Model) updateHiddenNodes(prevH []ag.Node, prevC []ag.Node, prevG ag.Node) ([]ag.Node, []ag.Node) {
	var wg sync.WaitGroup
	n := len(prevH)
	wg.Add(n)
	h := make([]ag.Node, n)
	c := make([]ag.Node, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			h[i], c[i] = m.processNode(i, prevH, prevC, prevG)
		}(i)
	}
	wg.Wait()
	return h, c
}

func (m *Model) updateSentenceState(prevH []ag.Node, prevC []ag.Node, prevG ag.Node) (ag.Node, ag.Node) {
	g := m.Graph()
	n := len(prevH)
	avgH := g.Mean(prevH)
	fG := g.Sigmoid(nn.Affine(g, m.NonLocalSentCellGate.B, m.NonLocalSentCellGate.W, prevG, m.NonLocalSentCellGate.U, avgH))
	oG := g.Sigmoid(nn.Affine(g, m.NonLocalSentOutputGate.B, m.NonLocalSentOutputGate.W, prevG, m.NonLocalSentOutputGate.U, avgH))

	hG := make([]ag.Node, n)
	gG := nn.Affine(g, m.NonLocalInputGate.B, m.NonLocalInputGate.W, prevG)
	for i := 0; i < n; i++ {
		hG[i] = g.Sigmoid(g.Add(gG, g.Mul(m.NonLocalInputGate.U, prevH[i])))
	}

	var sum ag.Node
	for i := 0; i < n; i++ {
		sum = g.Add(sum, g.Prod(hG[i], prevC[i]))
	}

	cg := g.Add(g.Prod(fG, prevG), sum)
	gt := g.Prod(oG, g.Tanh(cg))
	return gt, cg
}
