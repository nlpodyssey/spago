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
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Config provides configuration settings for a BLS Model.
type Config[T mat.DType] struct {
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
	FeaturesDropout              T
	EnhancedNodesDropout         T
}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config[T]
	Wz []nn.Param[T] `spago:"type:weights"`
	Bz []nn.Param[T] `spago:"type:biases"`
	Wh nn.Param[T]   `spago:"type:weights"`
	Bh nn.Param[T]   `spago:"type:biases"`
	W  nn.Param[T]   `spago:"type:weights"`
	B  nn.Param[T]   `spago:"type:biases"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](c Config[T]) *Model[T] {
	length := c.NumOfFeatures
	wz := make([]nn.Param[T], length)
	bz := make([]nn.Param[T], length)
	for i := 0; i < length; i++ {
		wz[i] = nn.NewParam[T](mat.NewEmptyDense[T](c.FeaturesSize, c.InputSize), nn.RequiresGrad[T](!c.KeepFeaturesParamsFixed))
		bz[i] = nn.NewParam[T](mat.NewEmptyVecDense[T](c.FeaturesSize), nn.RequiresGrad[T](!c.KeepFeaturesParamsFixed))
	}
	return &Model[T]{
		Config: c,
		Wz:     wz,
		Bz:     bz,
		Wh:     nn.NewParam[T](mat.NewEmptyDense[T](c.EnhancedNodesSize, c.NumOfFeatures*c.FeaturesSize), nn.RequiresGrad[T](!c.KeepEnhancedNodesParamsFixed)),
		Bh:     nn.NewParam[T](mat.NewEmptyVecDense[T](c.EnhancedNodesSize), nn.RequiresGrad[T](!c.KeepEnhancedNodesParamsFixed)),
		W:      nn.NewParam[T](mat.NewEmptyDense[T](c.OutputSize, c.NumOfFeatures*c.FeaturesSize+c.EnhancedNodesSize)),
		B:      nn.NewParam[T](mat.NewEmptyVecDense[T](c.OutputSize)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = m.forward(x)
	}
	return ys
}

func (m *Model[T]) forward(x ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	z := m.useFeaturesDropout(m.featuresMapping(x))
	h := m.useEnhancedNodesDropout(g.Invoke(m.EnhancedNodesActivation, nn.Affine[T](g, m.Bh, m.Wh, z)))
	y := g.Invoke(m.OutputActivation, nn.Affine[T](g, m.B, m.W, g.Concat([]ag.Node[T]{z, h}...)))
	return y
}

func (m *Model[T]) featuresMapping(x ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	z := make([]ag.Node[T], m.NumOfFeatures)
	for i := range z {
		z[i] = nn.Affine[T](g, m.Bz[i], m.Wz[i], x)
	}
	return g.Invoke(m.FeaturesActivation, g.Concat(z...))
}

func (m *Model[T]) useFeaturesDropout(x ag.Node[T]) ag.Node[T] {
	if m.Mode() == nn.Training && m.FeaturesDropout > 0.0 {
		return m.Graph().Dropout(x, m.FeaturesDropout)
	}
	return x
}

func (m *Model[T]) useEnhancedNodesDropout(x ag.Node[T]) ag.Node[T] {
	if m.Mode() == nn.Training && m.EnhancedNodesDropout > 0.0 {
		return m.Graph().Dropout(x, m.EnhancedNodesDropout)
	}
	return x
}
