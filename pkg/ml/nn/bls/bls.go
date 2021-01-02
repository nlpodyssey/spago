// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package bls provides an implementation of the Broad Learning System (BLS) described in "Broad Learning System:
An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture" by C. L. Philip Chen
and Zhulin Liu, 2017. (https://ieeexplore.ieee.org/document/7987745)

The "Model" contains only the inference part of the Broad Learning System.
The ridge regression approximation training is performed by the "BroadLearningAlgorithm".

Since the forward pass is built using the computational graph ("ag" a.k.a. auto-grad package), you can train the BLS
through the gradient-descent learning method, like other neural models, updating all the parameters and not just the
output weights as in the original implementation. Cool, isn't it? ;)
*/
package bls

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Config provides configuration settings for a BLS Model.
type Config struct {
	InputSize                    int
	FeaturesSize                 int
	NumOfFeatures                int
	EnhancedNodesSize            int
	OutputSize                   int
	FeaturesActivation           ag.OpName
	EnhancedNodesActivation      ag.OpName
	OutputActivation             ag.OpName
	KeepFeaturesParamsFixed      bool
	KeepEnhancedNodesParamsFixed bool
	FeaturesDropout              mat.Float
	EnhancedNodesDropout         mat.Float
}

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	Wz []nn.Param `spago:"type:weights"`
	Bz []nn.Param `spago:"type:biases"`
	Wh nn.Param   `spago:"type:weights"`
	Bh nn.Param   `spago:"type:biases"`
	W  nn.Param   `spago:"type:weights"`
	B  nn.Param   `spago:"type:biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(c Config) *Model {
	length := c.NumOfFeatures
	wz := make([]nn.Param, length)
	bz := make([]nn.Param, length)
	for i := 0; i < length; i++ {
		wz[i] = nn.NewParam(mat.NewEmptyDense(c.FeaturesSize, c.InputSize), nn.RequiresGrad(!c.KeepFeaturesParamsFixed))
		bz[i] = nn.NewParam(mat.NewEmptyVecDense(c.FeaturesSize), nn.RequiresGrad(!c.KeepFeaturesParamsFixed))
	}
	return &Model{
		Config: c,
		Wz:     wz,
		Bz:     bz,
		Wh:     nn.NewParam(mat.NewEmptyDense(c.EnhancedNodesSize, c.NumOfFeatures*c.FeaturesSize), nn.RequiresGrad(!c.KeepEnhancedNodesParamsFixed)),
		Bh:     nn.NewParam(mat.NewEmptyVecDense(c.EnhancedNodesSize), nn.RequiresGrad(!c.KeepEnhancedNodesParamsFixed)),
		W:      nn.NewParam(mat.NewEmptyDense(c.OutputSize, c.NumOfFeatures*c.FeaturesSize+c.EnhancedNodesSize)),
		B:      nn.NewParam(mat.NewEmptyVecDense(c.OutputSize)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = m.forward(x)
	}
	return ys
}

func (m *Model) forward(x ag.Node) ag.Node {
	g := m.Graph()
	z := m.useFeaturesDropout(m.featuresMapping(x))
	h := m.useEnhancedNodesDropout(g.Invoke(m.EnhancedNodesActivation, nn.Affine(g, m.Bh, m.Wh, z)))
	y := g.Invoke(m.OutputActivation, nn.Affine(g, m.B, m.W, g.Concat([]ag.Node{z, h}...)))
	return y
}

func (m *Model) featuresMapping(x ag.Node) ag.Node {
	g := m.Graph()
	z := make([]ag.Node, m.NumOfFeatures)
	for i := range z {
		z[i] = nn.Affine(g, m.Bz[i], m.Wz[i], x)
	}
	return g.Invoke(m.FeaturesActivation, g.Concat(z...))
}

func (m *Model) useFeaturesDropout(x ag.Node) ag.Node {
	if m.Mode() == nn.Training && m.FeaturesDropout > 0.0 {
		return m.Graph().Dropout(x, m.FeaturesDropout)
	}
	return x
}

func (m *Model) useEnhancedNodesDropout(x ag.Node) ag.Node {
	if m.Mode() == nn.Training && m.EnhancedNodesDropout > 0.0 {
		return m.Graph().Dropout(x, m.EnhancedNodesDropout)
	}
	return x
}
