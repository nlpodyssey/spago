// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package nru provides an implementation of the NRU (Non-Saturating Recurrent Units) recurrent network as described
in "Towards Non-Saturating Recurrent Units for Modelling Long-Term Dependencies" by Chandar et al., 2019.
(https://www.aaai.org/ojs/index.php/AAAI/article/view/4200/4078)

Unfortunately this implementation is extremely inefficient due to the lack of functionality in the auto-grad (ag)
package at the moment.
*/
package nru

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"log"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	SqrtMemK        int
	Wx              nn.Param `spago:"type:weights"`
	Wh              nn.Param `spago:"type:weights"`
	Wm              nn.Param `spago:"type:weights"`
	B               nn.Param `spago:"type:biases"`
	Whm2alpha       nn.Param `spago:"type:weights"`
	Bhm2alpha       nn.Param `spago:"type:biases"`
	Whm2alphaVec    nn.Param `spago:"type:weights"`
	Bhm2alphaVec    nn.Param `spago:"type:biases"`
	Whm2beta        nn.Param `spago:"type:weights"`
	Bhm2beta        nn.Param `spago:"type:biases"`
	Whm2betaVec     nn.Param `spago:"type:weights"`
	Bhm2betaVec     nn.Param `spago:"type:biases"`
	HiddenLayerNorm *layernorm.Model
}

// Config provides configuration settings for a NRU Model.
type Config struct {
	InputSize    int
	HiddenSize   int
	MemorySize   int
	K            int
	UseReLU      bool
	UseLayerNorm bool
	States       []*State `spago:"scope:processor"`
}

// State represent a state of the NRU recurrent network.
type State struct {
	Y      ag.Node
	Memory ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	if !isExactInt(mat.Sqrt(mat.Float(config.MemorySize * config.K))) {
		panic("nru: incompatible 'k' with 'memory size'")
	}
	sqrtMemK := int(mat.Sqrt(mat.Float(config.MemorySize * config.K)))

	return &Model{
		Wx:              nn.NewParam(mat.NewEmptyDense(config.HiddenSize, config.InputSize)),
		Wh:              nn.NewParam(mat.NewEmptyDense(config.HiddenSize, config.HiddenSize)),
		Wm:              nn.NewParam(mat.NewEmptyDense(config.HiddenSize, config.MemorySize)),
		B:               nn.NewParam(mat.NewEmptyVecDense(config.HiddenSize)),
		Whm2alpha:       nn.NewParam(mat.NewEmptyDense(config.K, config.MemorySize+config.HiddenSize)),
		Bhm2alpha:       nn.NewParam(mat.NewEmptyVecDense(config.K)),
		Whm2alphaVec:    nn.NewParam(mat.NewEmptyDense(2*sqrtMemK, config.MemorySize+config.HiddenSize)),
		Bhm2alphaVec:    nn.NewParam(mat.NewEmptyVecDense(2 * sqrtMemK)),
		Whm2beta:        nn.NewParam(mat.NewEmptyDense(config.K, config.MemorySize+config.HiddenSize)),
		Bhm2beta:        nn.NewParam(mat.NewEmptyVecDense(config.K)),
		Whm2betaVec:     nn.NewParam(mat.NewEmptyDense(2*sqrtMemK, config.MemorySize+config.HiddenSize)),
		Bhm2betaVec:     nn.NewParam(mat.NewEmptyVecDense(2 * sqrtMemK)),
		HiddenLayerNorm: layernorm.New(config.HiddenSize),
		SqrtMemK:        sqrtMemK,
	}
}

func isExactInt(val mat.Float) bool {
	return val == mat.Float(int(val))
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("nru: the initial state must be set before any input")
	}
	m.States = append(m.States, state)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := m.forward(x)
		m.States = append(m.States, s)
		ys[i] = s.Y
	}
	return ys
}

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (m *Model) LastState() *State {
	n := len(m.States)
	if n == 0 {
		return nil
	}
	return m.States[n-1]
}

func (m *Model) forward(x ag.Node) *State {
	g := m.Graph()
	yPrev, mPrev := m.getPrev()
	h := g.ReLU(m.optLayerNorm(nn.Affine(g, m.B, m.Wx, x, m.Wh, yPrev, m.Wm, mPrev)))
	hm := g.Concat(h, mPrev)
	addMemory := m.calcAddMemory(hm)
	forgetMemory := m.calcForgetMemory(hm)
	diffMemory := m.calcDiffMemory(addMemory, forgetMemory)
	return &State{
		Y:      h,
		Memory: g.Add(mPrev, diffMemory),
	}
}

func (m *Model) calcDiffMemory(addMemory, forgetMemory []ag.Node) ag.Node {
	g := m.Graph()
	diffMemory := make([]ag.Node, m.MemorySize)
	k := g.NewScalar(mat.Float(m.K))
	for j := 0; j < m.MemorySize; j++ {
		var sum ag.Node
		for i := 0; i < m.K; i++ {
			l := i*m.MemorySize + j
			sum = g.Add(sum, g.Sub(addMemory[l], forgetMemory[l]))
		}
		diffMemory[j] = g.Div(sum, k)
	}
	return g.Concat(diffMemory...)
}

func (m *Model) calcAddMemory(hm ag.Node) []ag.Node {
	g := m.Graph()
	alpha := nn.SeparateVec(g, m.optReLU(nn.Affine(g, m.Bhm2alpha, m.Whm2alpha, hm)))
	uAlpha := nn.SplitVec(g, nn.Affine(g, m.Bhm2alphaVec, m.Whm2alphaVec, hm), 2)
	uAlphaSecond := uAlpha[1]
	uAlphaFirst := uAlpha[0]
	vAlpha := make([]ag.Node, uAlphaFirst.Value().Size())
	for i := 0; i < m.SqrtMemK; i++ {
		vAlpha[i] = g.ProdScalar(uAlphaSecond, g.AtVec(uAlphaFirst, i))
	}
	vAlpha = m.optReLU2(vAlpha)
	vAlpha = normalization(g, vAlpha, 5)
	addMemory := make([]ag.Node, m.K*m.MemorySize)
	for i := 0; i < m.K; i++ {
		for j := 0; j < m.MemorySize; j++ {
			l := i*m.MemorySize + j
			ii := l / m.SqrtMemK
			jj := l % m.SqrtMemK
			addMemory[l] = g.Prod(alpha[i], g.AtVec(vAlpha[ii], jj))
		}
	}
	return addMemory
}

func (m *Model) calcForgetMemory(hm ag.Node) []ag.Node {
	g := m.Graph()
	beta := nn.SeparateVec(g, m.optReLU(nn.Affine(g, m.Bhm2beta, m.Whm2beta, hm)))
	uBeta := nn.SplitVec(g, nn.Affine(g, m.Bhm2betaVec, m.Whm2betaVec, hm), 2)
	uBetaSecond := uBeta[1]
	uBetaFirst := uBeta[0]
	vBeta := make([]ag.Node, uBetaFirst.Value().Size())
	for i := 0; i < m.SqrtMemK; i++ {
		vBeta[i] = g.ProdScalar(uBetaSecond, g.AtVec(uBetaFirst, i))
	}
	vBeta = m.optReLU2(vBeta)
	vBeta = normalization(g, vBeta, 5)
	forgetMemory := make([]ag.Node, m.K*m.MemorySize)
	for i := 0; i < m.K; i++ {
		for j := 0; j < m.MemorySize; j++ {
			l := i*m.MemorySize + j
			ii := l / m.SqrtMemK
			jj := l % m.SqrtMemK
			forgetMemory[l] = g.Prod(beta[i], g.AtVec(vBeta[ii], jj))
		}
	}
	return forgetMemory
}

func (m *Model) getPrev() (yPrev, mPrev ag.Node) {
	prev := m.LastState()
	if prev != nil {
		yPrev = prev.Y
		mPrev = prev.Memory
	} else {
		yPrev = m.Graph().NewVariable(mat.NewEmptyVecDense(m.HiddenSize), false)
		mPrev = m.Graph().NewVariable(mat.NewEmptyVecDense(m.MemorySize), false)
	}
	return
}

func (m *Model) optLayerNorm(x ag.Node) ag.Node {
	if m.UseLayerNorm {
		return nn.ToNode(m.HiddenLayerNorm.Forward(x))
	}
	return x
}

func (m *Model) optReLU(x ag.Node) ag.Node {
	if m.UseReLU {
		return m.Graph().ReLU(x)
	}
	return x
}

func (m *Model) optReLU2(xs []ag.Node) []ag.Node {
	if m.UseReLU {
		return ag.Map(func(x ag.Node) ag.Node { return m.Graph().ReLU(x) }, xs)
	}
	return xs
}

// TODO: improve performance and clean code
func normalization(g *ag.Graph, xs []ag.Node, p int) []ag.Node {
	dim0 := len(xs)
	dim1 := xs[0].Value().Size()
	tmp := make([][]ag.Node, dim0)
	for i := 0; i < dim0; i++ {
		tmp[i] = make([]ag.Node, dim1)
	}
	eps := g.NewScalar(1e-10)
	for j := 0; j < dim1; j++ {
		vec := make([]ag.Node, dim0)
		for i := 0; i < dim0; i++ {
			vec[i] = g.AtVec(xs[i], j)
		}
		norm := g.Add(pNorm(g, vec, p), eps)
		for i := 0; i < dim0; i++ {
			tmp[i][j] = g.DivScalar(vec[i], norm)
		}
	}
	ys := make([]ag.Node, dim0)
	for i := 0; i < dim0; i++ {
		ys[i] = g.Concat(tmp[i]...)
	}
	return ys
}

func pNorm(g *ag.Graph, xs []ag.Node, p int) ag.Node {
	var sum ag.Node
	for _, x := range xs {
		sum = g.Add(sum, g.Pow(g.Abs(x), mat.Float(p)))
	}
	return g.Pow(sum, 1.0/mat.Float(p))
}
