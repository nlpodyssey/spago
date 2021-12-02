// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gbst provides an implementation of the GBST (gradient-based subword tokenization)
// module from the Charformer paper (https://arxiv.org/abs/2106.12672).
// It automatically learns latent sub-words representations from characters in a data-driven fashion.
package gbst

import (
	"encoding/gob"
	"math"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution1d"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	Conv     []*convolution1d.Model
	Proj     []*convolution1d.Model
	Scorer   *linear.Model
	PadUntil int
}

// Config provides configuration settings for GBST (Gradient Based Subword Tokenization)
type Config struct {
	InputSize               int
	MaxBlockSize            int
	BlockSize               []int
	DownsampleFactor        int
	ScoreConsensusAttention bool
}

func init() {
	gob.Register(&Model{})
}

// New returns a new convolution Model, initialized according to the given configuration.
func New(config Config) *Model {
	if config.DownsampleFactor > config.MaxBlockSize {
		panic("gbst: downsample factor must be lower than maxiumum block size")
	}
	return &Model{
		Config: config,
		Conv: makeConvModels(config.InputSize, convolution1d.Config{
			KernelSizeX:    1,
			KernelSizeY:    config.MaxBlockSize,
			YStride:        1,
			InputChannels:  1,
			OutputChannels: 1,
			Mask:           nil,
			DepthWise:      false,
			Activation:     ag.OpIdentity,
		}),
		Proj: makeConvModels(config.InputSize, convolution1d.Config{
			KernelSizeX:    config.InputSize,
			KernelSizeY:    1,
			YStride:        1,
			InputChannels:  1,
			OutputChannels: 1,
			Mask:           nil,
			DepthWise:      false,
			Activation:     ag.OpIdentity,
		}),
		Scorer:   linear.New(config.InputSize, 1),
		PadUntil: lcm(config.BlockSize[0], config.BlockSize[1], config.BlockSize[2:len(config.BlockSize)]...),
	}
}

func makeConvModels(n int, config convolution1d.Config) []*convolution1d.Model {
	ms := make([]*convolution1d.Model, n)
	for i := 0; i < n; i++ {
		ms[i] = convolution1d.New(config)
	}
	return ms
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	l := len(xs)
	ys := make([]ag.Node, l)
	xs = m.padToMultiple(xs...)
	stackedIn := m.Graph().Stack(xs...)
	transposedStackedIn := m.Graph().T(stackedIn)
	stackedConvolvedXs := m.convolution(transposedStackedIn)
	convolvedEmbeddings := m.projection(stackedConvolvedXs, transposedStackedIn.Value().Rows())
	meanSequences := m.blocksMean(convolvedEmbeddings, l)
	scores := m.scorer(meanSequences, l)
	ys = m.weightSequence(meanSequences, scores, l)
	return m.downsample(ys)
}

// padToMultiple pads the sequence until lcm of blocks length
func (m *Model) padToMultiple(xs ...ag.Node) []ag.Node {
	n := nextDivisibleLength(len(xs), m.PadUntil)
	for i := 0; i < n; i++ {
		if i >= len(xs) {
			xs = append(xs, m.Graph().NewVariable(mat.NewEmptyVecDense(m.InputSize), false))
		}
	}
	return xs
}

// blockMean applies the average pooling of different size blocks. For example, considering blocks size 2
// out[0] = out[1] = average(in[0], in[1]);   out[2] = out[3] = average(in[2], in[3]) and so on.
func (m *Model) blockMean(sequence []ag.Node, outputSize, blockSize int) []ag.Node {
	out := make([]ag.Node, outputSize)
	l := len(sequence)
	for i := 0; i < l; i++ {
		if i < outputSize {
			if i%blockSize == 0 {
				if i+blockSize <= l {
					out[i] = m.Graph().Mean(sequence[i : i+blockSize])
				} else {
					out[i] = m.Graph().Mean(sequence[i:l])
				}
			} else {
				out[i] = m.Graph().Identity(out[i-1])
			}
		}
	}
	if l < outputSize {
		for k := l; k < outputSize; k++ {
			out[k] = m.Graph().Identity(out[l-1])
		}
	}

	return out
}

// seqMean applies the average pooling of the output sequence.
// This method, in contrast to the blockMean, returns a downsampled sequence.
// For example, considering blocks size 2
// out[0] =  average(in[0], in[1]);   out[1] = average(in[2], in[3]) and so on.
func (m *Model) seqMean(sequence []ag.Node, outputSize, blockSize int) []ag.Node {
	out := make([]ag.Node, outputSize)
	l := len(sequence)
	j := 0
	for i := 0; i < l; i++ {
		if i%blockSize == 0 {
			if i+blockSize <= l {
				out[j] = m.Graph().Mean(sequence[i : i+blockSize])
				j++
			} else {
				out[j] = m.Graph().Mean(sequence[i:l])
			}
		}
	}
	return out
}

// scorer is a parametrized linear function that produce a score for each candidate block.
func (m *Model) scorer(blocksSequence [][]ag.Node, length int) []ag.Node {
	scores := make([]ag.Node, length)
	for i := 0; i < length; i++ {
		ff := make([]ag.Node, len(m.BlockSize))
		for j, seq := range blocksSequence {
			ff[j] = m.Scorer.Forward(m.Graph().T(seq[i]))[0]
		}
		s := m.Graph().Concat(ff...)
		scores[i] = m.Graph().Softmax(s)
	}
	if m.Config.ScoreConsensusAttention {
		return m.scoreWithConsensusAttention(scores, length)
	}
	return scores
}

func (m *Model) scoreWithConsensusAttention(scores []ag.Node, length int) []ag.Node {
	scoresAttention := make([]ag.Node, length)
	stackedScores := m.Graph().Stack(scores...)
	dotProd := m.Graph().Mul(stackedScores, m.Graph().T(stackedScores))
	for i := 0; i < length; i++ {
		row := m.Graph().RowView(dotProd, i)
		softmaxAttention := m.Graph().Softmax(row)
		scoresAttention[i] = m.Graph().T(m.Graph().Mul(m.Graph().T(softmaxAttention), stackedScores))
	}
	return scoresAttention
}

// weightSequence calculates the weighted sum between all blocks and their score.
func (m *Model) weightSequence(blocksSequence [][]ag.Node, scores []ag.Node, length int) []ag.Node {
	out := make([]ag.Node, length)
	for i := 0; i < length; i++ {
		sepScores := nn.SeparateVec(m.Graph(), scores[i])
		weightedScore := m.Graph().NewVariable(mat.NewEmptyVecDense(m.InputSize), true)
		for j, seq := range blocksSequence {
			weightedScore = m.Graph().Add(weightedScore,
				m.Graph().ProdScalar(seq[i], sepScores[j]))
		}
		out[i] = weightedScore
	}
	return out
}

// convolution applies 1d convolution through the sequence, for each character dimension.
func (m *Model) convolution(xs ag.Node) ag.Node {
	convolved := make([]ag.Node, xs.Value().Rows())
	for i := 0; i < xs.Value().Rows(); i++ {
		row := m.Graph().RowView(xs, i)
		convolved[i] = m.Conv[i].Forward(row)[0]
	}
	return m.Graph().Stack(convolved...)
}

// projection applies a projection after the convolution through the sequence, for each character dimension.
func (m *Model) projection(inStackedVectors ag.Node, length int) []ag.Node {
	projectedXs := make([]ag.Node, length)
	for i := 0; i < length; i++ {
		projectedXs[i] = m.Proj[i].Forward(inStackedVectors)[0]
	}
	stackedProjectedXs := m.Graph().Stack(projectedXs...)
	stackedProjectedXs = m.Graph().T(stackedProjectedXs)
	convolvedEmbeddings := make([]ag.Node, stackedProjectedXs.Value().Rows())
	for i := range convolvedEmbeddings {
		convolvedEmbeddings[i] = m.Graph().RowView(stackedProjectedXs, i)
	}
	return convolvedEmbeddings
}

// blocksMean calculates the average pooling for each block of length 1 .. m.Blocksize, for the sequence of length l
func (m *Model) blocksMean(convolvedEmbeddings []ag.Node, length int) [][]ag.Node {
	meanSequences := make([][]ag.Node, len(m.BlockSize))
	meanSequences[0] = make([]ag.Node, length)
	maxLen := len(convolvedEmbeddings)
	for i := 0; i < length; i++ {
		if i < maxLen {
			meanSequences[0][i] = m.Graph().Identity(convolvedEmbeddings[i])
		} else {
			meanSequences[0][i] = m.Graph().Identity(convolvedEmbeddings[maxLen-1])
		}
	}
	for i := 1; i < len(meanSequences); i++ {
		meanSequences[i] = m.blockMean(convolvedEmbeddings, length, m.BlockSize[i])
	}
	return meanSequences
}

// downsample reduces the sequence by the downsample factor
func (m *Model) downsample(xs []ag.Node) []ag.Node {
	if m.DownsampleFactor < 2 {
		return xs
	}
	return m.seqMean(xs, int(math.Ceil(float64(len(xs))/float64(m.DownsampleFactor))), m.DownsampleFactor)
}
