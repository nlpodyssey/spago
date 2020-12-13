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
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"log"
	"math"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Config
	SqrtMemK        int
	Wx              *nn.Param `type:"weights"`
	Wh              *nn.Param `type:"weights"`
	Wm              *nn.Param `type:"weights"`
	B               *nn.Param `type:"biases"`
	Whm2alpha       *nn.Param `type:"weights"`
	Bhm2alpha       *nn.Param `type:"biases"`
	Whm2alphaVec    *nn.Param `type:"weights"`
	Bhm2alphaVec    *nn.Param `type:"biases"`
	Whm2beta        *nn.Param `type:"weights"`
	Bhm2beta        *nn.Param `type:"biases"`
	Whm2betaVec     *nn.Param `type:"weights"`
	Bhm2betaVec     *nn.Param `type:"biases"`
	HiddenLayerNorm *layernorm.Model
}

type Config struct {
	InputSize    int
	HiddenSize   int
	MemorySize   int
	K            int
	UseReLU      bool
	UseLayerNorm bool
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	if !isExactInt(math.Sqrt(float64(config.MemorySize * config.K))) {
		panic("nru: incompatible 'k' with 'memory size'")
	}
	sqrtMemK := int(math.Sqrt(float64(config.MemorySize * config.K)))

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

func isExactInt(val float64) bool {
	return val == float64(int(val))
}

type State struct {
	Y      ag.Node
	Memory ag.Node
}

type Processor struct {
	nn.BaseProcessor
	Config
	SqrtMemK        int
	wx              ag.Node
	wh              ag.Node
	wm              ag.Node
	b               ag.Node
	whm2alpha       ag.Node
	bhm2alpha       ag.Node
	whm2alphaVec    ag.Node
	bhm2alphaVec    ag.Node
	whm2beta        ag.Node
	bhm2beta        ag.Node
	whm2betaVec     ag.Node
	bhm2betaVec     ag.Node
	hiddenLayerNorm nn.Processor
	States          []*State
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		Config:          m.Config,
		SqrtMemK:        m.SqrtMemK,
		wx:              g.NewWrap(m.Wx),
		wh:              g.NewWrap(m.Wh),
		wm:              g.NewWrap(m.Wm),
		b:               g.NewWrap(m.B),
		whm2alpha:       g.NewWrap(m.Whm2alpha),
		bhm2alpha:       g.NewWrap(m.Bhm2alpha),
		whm2alphaVec:    g.NewWrap(m.Whm2alphaVec),
		bhm2alphaVec:    g.NewWrap(m.Bhm2alphaVec),
		whm2beta:        g.NewWrap(m.Whm2beta),
		bhm2beta:        g.NewWrap(m.Bhm2beta),
		whm2betaVec:     g.NewWrap(m.Whm2betaVec),
		bhm2betaVec:     g.NewWrap(m.Bhm2betaVec),
		hiddenLayerNorm: m.HiddenLayerNorm.NewProc(ctx),
		States:          nil,
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("nru: the initial state must be set before any input")
	}
	p.States = append(p.States, state)
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := p.forward(x)
		p.States = append(p.States, s)
		ys[i] = s.Y
	}
	return ys
}

func (p *Processor) LastState() *State {
	n := len(p.States)
	if n == 0 {
		return nil
	}
	return p.States[n-1]
}

func (p *Processor) forward(x ag.Node) *State {
	g := p.Graph
	yPrev, mPrev := p.getPrev()
	h := g.ReLU(p.optLayerNorm(nn.Affine(g, p.b, p.wx, x, p.wh, yPrev, p.wm, mPrev)))
	hm := g.Concat(h, mPrev)
	addMemory := p.calcAddMemory(hm)
	forgetMemory := p.calcForgetMemory(hm)
	diffMemory := p.calcDiffMemory(addMemory, forgetMemory)
	return &State{
		Y:      h,
		Memory: g.Add(mPrev, diffMemory),
	}
}

func (p *Processor) calcDiffMemory(addMemory, forgetMemory []ag.Node) ag.Node {
	g := p.Graph
	diffMemory := make([]ag.Node, p.MemorySize)
	k := g.NewScalar(float64(p.K))
	for j := 0; j < p.MemorySize; j++ {
		var sum ag.Node
		for i := 0; i < p.K; i++ {
			l := i*p.MemorySize + j
			sum = g.Add(sum, g.Sub(addMemory[l], forgetMemory[l]))
		}
		diffMemory[j] = g.Div(sum, k)
	}
	return g.Concat(diffMemory...)
}

func (p *Processor) calcAddMemory(hm ag.Node) []ag.Node {
	g := p.Graph
	alpha := nn.SeparateVec(g, p.optReLU(nn.Affine(g, p.bhm2alpha, p.whm2alpha, hm)))
	uAlpha := nn.SplitVec(g, nn.Affine(g, p.bhm2alphaVec, p.whm2alphaVec, hm), 2)
	uAlphaSecond := uAlpha[1]
	uAlphaFirst := uAlpha[0]
	vAlpha := make([]ag.Node, uAlphaFirst.Value().Size())
	for i := 0; i < p.SqrtMemK; i++ {
		vAlpha[i] = g.ProdScalar(uAlphaSecond, g.AtVec(uAlphaFirst, i))
	}
	vAlpha = p.optReLU2(vAlpha)
	vAlpha = normalization(g, vAlpha, 5)
	addMemory := make([]ag.Node, p.K*p.MemorySize)
	for i := 0; i < p.K; i++ {
		for j := 0; j < p.MemorySize; j++ {
			l := i*p.MemorySize + j
			ii := l / p.SqrtMemK
			jj := l % p.SqrtMemK
			addMemory[l] = g.Prod(alpha[i], g.AtVec(vAlpha[ii], jj))
		}
	}
	return addMemory
}

func (p *Processor) calcForgetMemory(hm ag.Node) []ag.Node {
	g := p.Graph
	beta := nn.SeparateVec(g, p.optReLU(nn.Affine(g, p.bhm2beta, p.whm2beta, hm)))
	uBeta := nn.SplitVec(g, nn.Affine(g, p.bhm2betaVec, p.whm2betaVec, hm), 2)
	uBetaSecond := uBeta[1]
	uBetaFirst := uBeta[0]
	vBeta := make([]ag.Node, uBetaFirst.Value().Size())
	for i := 0; i < p.SqrtMemK; i++ {
		vBeta[i] = g.ProdScalar(uBetaSecond, g.AtVec(uBetaFirst, i))
	}
	vBeta = p.optReLU2(vBeta)
	vBeta = normalization(g, vBeta, 5)
	forgetMemory := make([]ag.Node, p.K*p.MemorySize)
	for i := 0; i < p.K; i++ {
		for j := 0; j < p.MemorySize; j++ {
			l := i*p.MemorySize + j
			ii := l / p.SqrtMemK
			jj := l % p.SqrtMemK
			forgetMemory[l] = g.Prod(beta[i], g.AtVec(vBeta[ii], jj))
		}
	}
	return forgetMemory
}

func (p *Processor) getPrev() (yPrev, mPrev ag.Node) {
	prev := p.LastState()
	if prev != nil {
		yPrev = prev.Y
		mPrev = prev.Memory
	} else {
		yPrev = p.Graph.NewVariable(mat.NewEmptyVecDense(p.HiddenSize), false)
		mPrev = p.Graph.NewVariable(mat.NewEmptyVecDense(p.MemorySize), false)
	}
	return
}

func (p *Processor) optLayerNorm(x ag.Node) ag.Node {
	if p.UseLayerNorm {
		return p.hiddenLayerNorm.Forward(x)[0]
	}
	return x
}

func (p *Processor) optReLU(x ag.Node) ag.Node {
	if p.UseReLU {
		return p.Graph.ReLU(x)
	}
	return x
}

func (p *Processor) optReLU2(xs []ag.Node) []ag.Node {
	if p.UseReLU {
		return ag.Map(func(x ag.Node) ag.Node { return p.Graph.ReLU(x) }, xs)
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
		sum = g.Add(sum, g.Pow(g.Abs(x), float64(p)))
	}
	return g.Pow(sum, 1.0/float64(p))
}
