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
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// TODO(1): code refactoring using a structure to maintain states.
// TODO(2): use a gradient policy (i.e. reinforcement learning) to increase the context with dynamic skip connections.

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Config                 Config
	InputGate              *HyperLinear4
	LeftCellGate           *HyperLinear4
	RightCellGate          *HyperLinear4
	CellGate               *HyperLinear4
	SentCellGate           *HyperLinear4
	OutputGate             *HyperLinear4
	InputActivation        *HyperLinear4
	NonLocalSentCellGate   *HyperLinear3
	NonLocalInputGate      *HyperLinear3
	NonLocalSentOutputGate *HyperLinear3
	StartH                 *nn.Param
	EndH                   *nn.Param
	InitValue              *nn.Param
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
	nn.Module
	W *nn.Param
	U *nn.Param
	V *nn.Param
	B *nn.Param
}

// HyperLinear3 groups multiple params for an affine transformation.
type HyperLinear3 struct {
	nn.Module
	W *nn.Param
	U *nn.Param
	B *nn.Param
}

// State contains nodes used during the forward step.
type State struct {
	xUi []mat.Tensor
	xUl []mat.Tensor
	xUr []mat.Tensor
	xUf []mat.Tensor
	xUs []mat.Tensor
	xUo []mat.Tensor
	xUu []mat.Tensor

	ViPrevG mat.Tensor
	VlPrevG mat.Tensor
	VrPrevG mat.Tensor
	VfPrevG mat.Tensor
	VsPrevG mat.Tensor
	VoPrevG mat.Tensor
	VuPrevG mat.Tensor
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](config Config) *Model {
	in, out := config.InputSize, config.OutputSize
	return &Model{
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
		StartH:                 nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
		EndH:                   nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
		InitValue:              nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
	}
}

func newGate4[T float.DType](in, out int) *HyperLinear4 {
	return &HyperLinear4{
		W: nn.NewParam(mat.NewDense[T](mat.WithShape(out, out*windowSize))),
		U: nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		V: nn.NewParam(mat.NewDense[T](mat.WithShape(out, out))),
		B: nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
	}
}

func newGate3[T float.DType](size int) *HyperLinear3 {
	return &HyperLinear3{
		W: nn.NewParam(mat.NewDense[T](mat.WithShape(size, size))),
		U: nn.NewParam(mat.NewDense[T](mat.WithShape(size, size))),
		B: nn.NewParam(mat.NewDense[T](mat.WithShape(size))),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...mat.Tensor) []mat.Tensor {
	steps := m.Config.Steps
	n := len(xs)
	h := make([][]mat.Tensor, steps)
	c := make([][]mat.Tensor, steps)
	g := make([]mat.Tensor, steps)
	cg := make([]mat.Tensor, steps)
	h[0] = make([]mat.Tensor, n)
	c[0] = make([]mat.Tensor, n)

	g[0] = m.InitValue
	cg[0] = m.InitValue
	for i := 0; i < n; i++ {
		h[0][i] = m.InitValue
		c[0][i] = m.InitValue
	}

	s := &State{}
	m.computeUx(s, xs) // the result is shared among all steps
	for t := 1; t < m.Config.Steps; t++ {
		m.computeVg(s, g[t-1]) // the result is shared among all nodes of the same step
		h[t], c[t] = m.updateHiddenNodes(s, h[t-1], c[t-1], g[t-1])
		g[t], cg[t] = m.updateSentenceState(h[t-1], c[t-1], g[t-1])
	}

	return h[len(h)-1]
}

func (m *Model) computeUx(s *State, xs []mat.Tensor) {
	n := len(xs)
	s.xUi = make([]mat.Tensor, n)
	s.xUl = make([]mat.Tensor, n)
	s.xUr = make([]mat.Tensor, n)
	s.xUf = make([]mat.Tensor, n)
	s.xUs = make([]mat.Tensor, n)
	s.xUo = make([]mat.Tensor, n)
	s.xUu = make([]mat.Tensor, n)

	for i := 0; i < n; i++ {
		s.xUi[i] = ag.Mul(m.InputGate.U, xs[i])
		s.xUl[i] = ag.Mul(m.LeftCellGate.U, xs[i])
		s.xUr[i] = ag.Mul(m.RightCellGate.U, xs[i])
		s.xUf[i] = ag.Mul(m.CellGate.U, xs[i])
		s.xUs[i] = ag.Mul(m.SentCellGate.U, xs[i])
		s.xUo[i] = ag.Mul(m.OutputGate.U, xs[i])
		s.xUu[i] = ag.Mul(m.InputActivation.U, xs[i])
	}
}

func (m *Model) computeVg(s *State, prevG mat.Tensor) {
	s.ViPrevG = ag.Mul(m.InputGate.V, prevG)
	s.VlPrevG = ag.Mul(m.LeftCellGate.V, prevG)
	s.VrPrevG = ag.Mul(m.RightCellGate.V, prevG)
	s.VfPrevG = ag.Mul(m.CellGate.V, prevG)
	s.VsPrevG = ag.Mul(m.SentCellGate.V, prevG)
	s.VoPrevG = ag.Mul(m.OutputGate.V, prevG)
	s.VuPrevG = ag.Mul(m.InputActivation.U, prevG)
}

func (m *Model) updateHiddenNodes(s *State, prevH []mat.Tensor, prevC []mat.Tensor, prevG mat.Tensor) ([]mat.Tensor, []mat.Tensor) {
	n := len(prevH)
	h := make([]mat.Tensor, n)
	c := make([]mat.Tensor, n)
	for i := 0; i < n; i++ {
		h[i], c[i] = m.processNode(s, i, prevH, prevC, prevG)
	}
	return h, c
}

func (m *Model) updateSentenceState(prevH []mat.Tensor, prevC []mat.Tensor, prevG mat.Tensor) (mat.Tensor, mat.Tensor) {
	n := len(prevH)
	avgH := ag.Mean(prevH)
	fG := ag.Sigmoid(ag.Affine(m.NonLocalSentCellGate.B, m.NonLocalSentCellGate.W, prevG, m.NonLocalSentCellGate.U, avgH))
	oG := ag.Sigmoid(ag.Affine(m.NonLocalSentOutputGate.B, m.NonLocalSentOutputGate.W, prevG, m.NonLocalSentOutputGate.U, avgH))

	hG := make([]mat.Tensor, n)
	gG := ag.Affine(m.NonLocalInputGate.B, m.NonLocalInputGate.W, prevG)
	for i := 0; i < n; i++ {
		hG[i] = ag.Sigmoid(ag.Add(gG, ag.Mul(m.NonLocalInputGate.U, prevH[i])))
	}

	var sum mat.Tensor
	for i := 0; i < n; i++ {
		sum = ag.Add(sum, ag.Prod(hG[i], prevC[i]))
	}

	cg := ag.Add(ag.Prod(fG, prevG), sum)
	gt := ag.Prod(oG, ag.Tanh(cg))
	return gt, cg
}

func (m *Model) processNode(s *State, i int, prevH []mat.Tensor, prevC []mat.Tensor, prevG mat.Tensor) (h mat.Tensor, c mat.Tensor) {
	n := len(prevH)
	first := 0
	last := n - 1
	j := i - 1
	k := i + 1

	var prevHj, prevCj mat.Tensor
	if j < first {
		prevHj, prevCj = m.StartH, m.StartH
	} else {
		prevHj, prevCj = prevH[j], prevC[j]
	}

	var prevHk, prevCk mat.Tensor
	if k > last {
		prevHk, prevCk = m.EndH, m.EndH
	} else {
		prevHk, prevCk = prevH[k], prevC[k]
	}

	context := ag.Concat(prevHj, prevH[i], prevHk)
	iG := ag.Sigmoid(ag.Sum(m.InputGate.B, ag.Mul(m.InputGate.W, context), s.ViPrevG, s.xUi[i]))
	lG := ag.Sigmoid(ag.Sum(m.LeftCellGate.B, ag.Mul(m.LeftCellGate.W, context), s.VlPrevG, s.xUl[i]))
	rG := ag.Sigmoid(ag.Sum(m.RightCellGate.B, ag.Mul(m.RightCellGate.W, context), s.VrPrevG, s.xUr[i]))
	fG := ag.Sigmoid(ag.Sum(m.CellGate.B, ag.Mul(m.CellGate.W, context), s.VfPrevG, s.xUf[i]))
	sG := ag.Sigmoid(ag.Sum(m.SentCellGate.B, ag.Mul(m.SentCellGate.W, context), s.VsPrevG, s.xUs[i]))
	oG := ag.Sigmoid(ag.Sum(m.OutputGate.B, ag.Mul(m.OutputGate.W, context), s.VoPrevG, s.xUo[i]))
	uG := ag.Tanh(ag.Sum(m.InputActivation.B, ag.Mul(m.InputActivation.W, context), s.VuPrevG, s.xUu[i]))
	c1 := ag.Prod(lG, prevCj)
	c2 := ag.Prod(fG, prevC[i])
	c3 := ag.Prod(rG, prevCk)
	c4 := ag.Prod(sG, prevG)
	c5 := ag.Prod(iG, uG)
	c = ag.Sum(c1, c2, c3, c4, c5)
	h = ag.Prod(oG, ag.Tanh(c))

	return
}
