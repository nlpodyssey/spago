// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Implementation of the NRU (Non-Saturating Recurrent Units) recurrent network as described in "Towards Non-Saturating
Recurrent Units for Modelling Long-Term Dependencies" by Chandar et al., 2019.
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
	"io"
	"log"
	"math"
)

var _ nn.Model = &Model{}

type Model struct {
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
	memorySize      int
	hiddenSize      int
	sqrtMemK        int
	k               int
	useReLU         bool
	useLayerNorm    bool
}

func New(inputSize, hiddenSize, memorySize, k int, useReLU bool, useLayerNorm bool) *Model {
	if !isExactInt(math.Sqrt(float64(memorySize * k))) {
		panic("nru: incompatible 'k' with 'memory size'")
	}
	sqrtMemK := int(math.Sqrt(float64(memorySize * k)))

	return &Model{
		Wx:              nn.NewParam(mat.NewEmptyDense(hiddenSize, inputSize)),
		Wh:              nn.NewParam(mat.NewEmptyDense(hiddenSize, hiddenSize)),
		Wm:              nn.NewParam(mat.NewEmptyDense(hiddenSize, memorySize)),
		B:               nn.NewParam(mat.NewEmptyVecDense(hiddenSize)),
		Whm2alpha:       nn.NewParam(mat.NewEmptyDense(k, memorySize+hiddenSize)),
		Bhm2alpha:       nn.NewParam(mat.NewEmptyVecDense(k)),
		Whm2alphaVec:    nn.NewParam(mat.NewEmptyDense(2*sqrtMemK, memorySize+hiddenSize)),
		Bhm2alphaVec:    nn.NewParam(mat.NewEmptyVecDense(2 * sqrtMemK)),
		Whm2beta:        nn.NewParam(mat.NewEmptyDense(k, memorySize+hiddenSize)),
		Bhm2beta:        nn.NewParam(mat.NewEmptyVecDense(k)),
		Whm2betaVec:     nn.NewParam(mat.NewEmptyDense(2*sqrtMemK, memorySize+hiddenSize)),
		Bhm2betaVec:     nn.NewParam(mat.NewEmptyVecDense(2 * sqrtMemK)),
		HiddenLayerNorm: layernorm.New(hiddenSize),
		memorySize:      memorySize,
		hiddenSize:      hiddenSize,
		sqrtMemK:        sqrtMemK,
		k:               k,
		useReLU:         useReLU,
		useLayerNorm:    useLayerNorm,
	}
}

func isExactInt(val float64) bool {
	return val == float64(int(val))
}

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

type State struct {
	Y      ag.Node
	Memory ag.Node
}

type InitHidden struct {
	*State
}

var _ nn.Processor = &Processor{}

type Processor struct {
	opt             []interface{}
	model           *Model
	mode            nn.ProcessingMode
	g               *ag.Graph
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

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:           m,
		mode:            nn.Training,
		States:          nil,
		opt:             opt,
		g:               g,
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
		hiddenLayerNorm: m.HiddenLayerNorm.NewProc(g),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	for _, t := range opt {
		switch t := t.(type) {
		case InitHidden:
			p.States = append(p.States, t.State)
		default:
			log.Fatal("nru: invalid init option")
		}
	}
}

func (p *Processor) Model() nn.Model         { return p.model }
func (p *Processor) Graph() *ag.Graph        { return p.g }
func (p *Processor) RequiresFullSeq() bool   { return false }
func (p *Processor) Mode() nn.ProcessingMode { return p.mode }

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.mode = mode
	p.hiddenLayerNorm.SetMode(mode)
}

func (p *Processor) Reset() {
	p.States = nil
	p.init(p.opt)
}

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
	yPrev, mPrev := p.getPrev()
	h := p.g.ReLU(p.optLayerNorm(nn.Affine(p.g, p.b, p.wx, x, p.wh, yPrev, p.wm, mPrev)))
	hm := p.g.Concat(h, mPrev)
	addMemory := p.calcAddMemory(hm)
	forgetMemory := p.calcForgetMemory(hm)
	diffMemory := p.calcDiffMemory(addMemory, forgetMemory)
	return &State{
		Y:      h,
		Memory: p.g.Add(mPrev, diffMemory),
	}
}

func (p *Processor) calcDiffMemory(addMemory, forgetMemory []ag.Node) ag.Node {
	diffMemory := make([]ag.Node, p.model.memorySize)
	k := p.g.NewScalar(float64(p.model.k))
	for j := 0; j < p.model.memorySize; j++ {
		var sum ag.Node
		for i := 0; i < p.model.k; i++ {
			l := i*p.model.memorySize + j
			sum = p.g.Add(sum, p.g.Sub(addMemory[l], forgetMemory[l]))
		}
		diffMemory[j] = p.g.Div(sum, k)
	}
	return p.g.Concat(diffMemory...)
}

func (p *Processor) calcAddMemory(hm ag.Node) []ag.Node {
	alpha := nn.SeparateVec(p.g, p.optReLU(nn.Affine(p.g, p.bhm2alpha, p.whm2alpha, hm)))
	uAlpha := nn.SplitVec(p.g, nn.Affine(p.g, p.bhm2alphaVec, p.whm2alphaVec, hm), 2)
	vAlpha := make([]ag.Node, uAlpha[0].Value().Size())
	for i := 0; i < p.model.sqrtMemK; i++ {
		vAlpha[i] = p.g.ProdScalar(uAlpha[1], p.g.AtVec(uAlpha[0], i))
	}
	vAlpha = p.optReLU2(vAlpha)
	vAlpha = normalization(p.g, vAlpha, 5)
	addMemory := make([]ag.Node, p.model.k*p.model.memorySize)
	for i := 0; i < p.model.k; i++ {
		for j := 0; j < p.model.memorySize; j++ {
			l := i*p.model.memorySize + j
			ii := l / p.model.sqrtMemK
			jj := l % p.model.sqrtMemK
			addMemory[l] = p.g.Prod(alpha[i], p.g.AtVec(vAlpha[ii], jj))
		}
	}
	return addMemory
}

func (p *Processor) calcForgetMemory(hm ag.Node) []ag.Node {
	beta := nn.SeparateVec(p.g, p.optReLU(nn.Affine(p.g, p.bhm2beta, p.whm2beta, hm)))
	uBeta := nn.SplitVec(p.g, nn.Affine(p.g, p.bhm2betaVec, p.whm2betaVec, hm), 2)
	vBeta := make([]ag.Node, uBeta[0].Value().Size())
	for i := 0; i < p.model.sqrtMemK; i++ {
		vBeta[i] = p.g.ProdScalar(uBeta[1], p.g.AtVec(uBeta[0], i))
	}
	vBeta = p.optReLU2(vBeta)
	vBeta = normalization(p.g, vBeta, 5)
	forgetMemory := make([]ag.Node, p.model.k*p.model.memorySize)
	for i := 0; i < p.model.k; i++ {
		for j := 0; j < p.model.memorySize; j++ {
			l := i*p.model.memorySize + j
			ii := l / p.model.sqrtMemK
			jj := l % p.model.sqrtMemK
			forgetMemory[l] = p.g.Prod(beta[i], p.g.AtVec(vBeta[ii], jj))
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
		yPrev = p.g.NewVariable(mat.NewEmptyVecDense(p.model.hiddenSize), false)
		mPrev = p.g.NewVariable(mat.NewEmptyVecDense(p.model.memorySize), false)
	}
	return
}

func (p *Processor) optLayerNorm(x ag.Node) ag.Node {
	if p.model.useLayerNorm {
		return p.hiddenLayerNorm.Forward(x)[0]
	} else {
		return x
	}
}

func (p *Processor) optReLU(x ag.Node) ag.Node {
	if p.model.useReLU {
		return p.g.ReLU(x)
	} else {
		return x
	}
}

func (p *Processor) optReLU2(xs []ag.Node) []ag.Node {
	if p.model.useReLU {
		ys := make([]ag.Node, len(xs))
		for i, x := range xs {
			ys[i] = p.g.ReLU(x)
		}
		return ys
	} else {
		return xs
	}
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
