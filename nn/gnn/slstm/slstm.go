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
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"sync"
)

var _ nn.Model[float32] = &Model[float32]{}

// TODO(1): code refactoring using a structure to maintain states.
// TODO(2): use a gradient policy (i.e. reinforcement learning) to increase the context with dynamic skip connections.

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config                 Config
	InputGate              *HyperLinear4[T] `spago:"type:params"`
	LeftCellGate           *HyperLinear4[T] `spago:"type:params"`
	RightCellGate          *HyperLinear4[T] `spago:"type:params"`
	CellGate               *HyperLinear4[T] `spago:"type:params"`
	SentCellGate           *HyperLinear4[T] `spago:"type:params"`
	OutputGate             *HyperLinear4[T] `spago:"type:params"`
	InputActivation        *HyperLinear4[T] `spago:"type:params"`
	NonLocalSentCellGate   *HyperLinear3[T] `spago:"type:params"`
	NonLocalInputGate      *HyperLinear3[T] `spago:"type:params"`
	NonLocalSentOutputGate *HyperLinear3[T] `spago:"type:params"`
	StartH                 nn.Param[T]      `spago:"type:weights"`
	EndH                   nn.Param[T]      `spago:"type:weights"`
	InitValue              nn.Param[T]      `spago:"type:weights"`
	Support                Support[T]       `spago:"scope:processor"`
}

// Config provides configuration settings for a Sentence-State LSTM Model.
type Config struct {
	InputSize  int
	OutputSize int
	Steps      int
}

const windowSize = 3 // the window is fixed in this implementation

// HyperLinear4 groups multiple params for an affine transformation.
type HyperLinear4[T mat.DType] struct {
	W nn.Param[T] `spago:"type:weights"`
	U nn.Param[T] `spago:"type:weights"`
	V nn.Param[T] `spago:"type:weights"`
	B nn.Param[T] `spago:"type:biases"`
}

// HyperLinear3 groups multiple params for an affine transformation.
type HyperLinear3[T mat.DType] struct {
	W nn.Param[T] `spago:"type:weights"`
	U nn.Param[T] `spago:"type:weights"`
	B nn.Param[T] `spago:"type:biases"`
}

// Support contains nodes used during the forward step.
type Support[T mat.DType] struct {
	// Shared among all steps
	xUi, xUl, xUr, xUf, xUs, xUo, xUu []ag.Node[T]
	// Shared among all nodes at the same step
	ViPrevG ag.Node[T]
	VlPrevG ag.Node[T]
	VrPrevG ag.Node[T]
	VfPrevG ag.Node[T]
	VsPrevG ag.Node[T]
	VoPrevG ag.Node[T]
	VuPrevG ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](config Config) *Model[T] {
	in, out := config.InputSize, config.OutputSize
	return &Model[T]{
		Config:                 config,
		InputGate:              newGate4[T](in, out),
		LeftCellGate:           newGate4[T](in, out),
		RightCellGate:          newGate4[T](in, out),
		CellGate:               newGate4[T](in, out),
		SentCellGate:           newGate4[T](in, out),
		OutputGate:             newGate4[T](in, out),
		InputActivation:        newGate4[T](in, out),
		NonLocalSentCellGate:   newGate3[T](out),
		NonLocalInputGate:      newGate3[T](out),
		NonLocalSentOutputGate: newGate3[T](out),
		StartH:                 nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		EndH:                   nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		InitValue:              nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
	}
}

func newGate4[T mat.DType](in, out int) *HyperLinear4[T] {
	return &HyperLinear4[T]{
		W: nn.NewParam[T](mat.NewEmptyDense[T](out, out*windowSize)),
		U: nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		V: nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		B: nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
	}
}

func newGate3[T mat.DType](size int) *HyperLinear3[T] {
	return &HyperLinear3[T]{
		W: nn.NewParam[T](mat.NewEmptyDense[T](size, size)),
		U: nn.NewParam[T](mat.NewEmptyDense[T](size, size)),
		B: nn.NewParam[T](mat.NewEmptyVecDense[T](size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	steps := m.Config.Steps
	n := len(xs)
	h := make([][]ag.Node[T], steps)
	c := make([][]ag.Node[T], steps)
	g := make([]ag.Node[T], steps)
	cg := make([]ag.Node[T], steps)
	h[0] = make([]ag.Node[T], n)
	c[0] = make([]ag.Node[T], n)

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

func (m *Model[T]) computeUx(xs []ag.Node[T]) {
	n := len(xs)
	m.Support.xUi = make([]ag.Node[T], n)
	m.Support.xUl = make([]ag.Node[T], n)
	m.Support.xUr = make([]ag.Node[T], n)
	m.Support.xUf = make([]ag.Node[T], n)
	m.Support.xUs = make([]ag.Node[T], n)
	m.Support.xUo = make([]ag.Node[T], n)
	m.Support.xUu = make([]ag.Node[T], n)

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			m.Support.xUi[i] = ag.Mul[T](m.InputGate.U, xs[i])
			m.Support.xUl[i] = ag.Mul[T](m.LeftCellGate.U, xs[i])
			m.Support.xUr[i] = ag.Mul[T](m.RightCellGate.U, xs[i])
			m.Support.xUf[i] = ag.Mul[T](m.CellGate.U, xs[i])
			m.Support.xUs[i] = ag.Mul[T](m.SentCellGate.U, xs[i])
			m.Support.xUo[i] = ag.Mul[T](m.OutputGate.U, xs[i])
			m.Support.xUu[i] = ag.Mul[T](m.InputActivation.U, xs[i])
		}(i)
	}
	wg.Wait()
}

func (m *Model[T]) computeVg(prevG ag.Node[T]) {
	var wg sync.WaitGroup
	wg.Add(7)
	for i := 0; i < 7; i++ {
		go func(i int) {
			defer wg.Done()
			switch i {
			case 0:
				m.Support.ViPrevG = ag.Mul[T](m.InputGate.V, prevG)
			case 1:
				m.Support.VlPrevG = ag.Mul[T](m.LeftCellGate.V, prevG)
			case 2:
				m.Support.VrPrevG = ag.Mul[T](m.RightCellGate.V, prevG)
			case 3:
				m.Support.VfPrevG = ag.Mul[T](m.CellGate.V, prevG)
			case 4:
				m.Support.VsPrevG = ag.Mul[T](m.SentCellGate.V, prevG)
			case 5:
				m.Support.VoPrevG = ag.Mul[T](m.OutputGate.V, prevG)
			case 6:
				m.Support.VuPrevG = ag.Mul[T](m.InputActivation.U, prevG)
			}
		}(i)
	}
	wg.Wait()
}

func (m *Model[T]) processNode(i int, prevH []ag.Node[T], prevC []ag.Node[T], prevG ag.Node[T]) (h ag.Node[T], c ag.Node[T]) {
	n := len(prevH)
	first := 0
	last := n - 1
	j := i - 1
	k := i + 1

	prevHj, prevCj := func() (ag.Node[T], ag.Node[T]) {
		if j < first {
			return m.StartH, m.StartH
		}
		return prevH[j], prevC[j]
	}()

	prevHk, prevCk := func() (ag.Node[T], ag.Node[T]) {
		if k > last {
			return m.EndH, m.EndH
		}
		return prevH[k], prevC[k]
	}()

	context := ag.Concat(prevHj, prevH[i], prevHk)
	iG := ag.Sigmoid(ag.Sum[T](m.InputGate.B, ag.Mul[T](m.InputGate.W, context), m.Support.ViPrevG, m.Support.xUi[i]))
	lG := ag.Sigmoid(ag.Sum[T](m.LeftCellGate.B, ag.Mul[T](m.LeftCellGate.W, context), m.Support.VlPrevG, m.Support.xUl[i]))
	rG := ag.Sigmoid(ag.Sum[T](m.RightCellGate.B, ag.Mul[T](m.RightCellGate.W, context), m.Support.VrPrevG, m.Support.xUr[i]))
	fG := ag.Sigmoid(ag.Sum[T](m.CellGate.B, ag.Mul[T](m.CellGate.W, context), m.Support.VfPrevG, m.Support.xUf[i]))
	sG := ag.Sigmoid(ag.Sum[T](m.SentCellGate.B, ag.Mul[T](m.SentCellGate.W, context), m.Support.VsPrevG, m.Support.xUs[i]))
	oG := ag.Sigmoid(ag.Sum[T](m.OutputGate.B, ag.Mul[T](m.OutputGate.W, context), m.Support.VoPrevG, m.Support.xUo[i]))
	uG := ag.Tanh(ag.Sum[T](m.InputActivation.B, ag.Mul[T](m.InputActivation.W, context), m.Support.VuPrevG, m.Support.xUu[i]))
	c1 := ag.Prod(lG, prevCj)
	c2 := ag.Prod(fG, prevC[i])
	c3 := ag.Prod(rG, prevCk)
	c4 := ag.Prod(sG, prevG)
	c5 := ag.Prod(iG, uG)
	c = ag.Sum(c1, c2, c3, c4, c5)
	h = ag.Prod(oG, ag.Tanh(c))
	return
}

func (m *Model[T]) updateHiddenNodes(prevH []ag.Node[T], prevC []ag.Node[T], prevG ag.Node[T]) ([]ag.Node[T], []ag.Node[T]) {
	var wg sync.WaitGroup
	n := len(prevH)
	wg.Add(n)
	h := make([]ag.Node[T], n)
	c := make([]ag.Node[T], n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			h[i], c[i] = m.processNode(i, prevH, prevC, prevG)
		}(i)
	}
	wg.Wait()
	return h, c
}

func (m *Model[T]) updateSentenceState(prevH []ag.Node[T], prevC []ag.Node[T], prevG ag.Node[T]) (ag.Node[T], ag.Node[T]) {
	n := len(prevH)
	avgH := ag.Mean(prevH)
	fG := ag.Sigmoid(ag.Affine[T](m.NonLocalSentCellGate.B, m.NonLocalSentCellGate.W, prevG, m.NonLocalSentCellGate.U, avgH))
	oG := ag.Sigmoid(ag.Affine[T](m.NonLocalSentOutputGate.B, m.NonLocalSentOutputGate.W, prevG, m.NonLocalSentOutputGate.U, avgH))

	hG := make([]ag.Node[T], n)
	gG := ag.Affine[T](m.NonLocalInputGate.B, m.NonLocalInputGate.W, prevG)
	for i := 0; i < n; i++ {
		hG[i] = ag.Sigmoid(ag.Add[T](gG, ag.Mul[T](m.NonLocalInputGate.U, prevH[i])))
	}

	var sum ag.Node[T]
	for i := 0; i < n; i++ {
		sum = ag.Add(sum, ag.Prod(hG[i], prevC[i]))
	}

	cg := ag.Add(ag.Prod(fG, prevG), sum)
	gt := ag.Prod(oG, ag.Tanh(cg))
	return gt, cg
}
