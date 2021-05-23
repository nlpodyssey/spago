// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

func TestModel(t *testing.T) {
	embeddingsDBPath := createTempDir(t, "gmlp_test_embeddings")
	defer deleteDir(t, embeddingsDBPath)

	model := New(Config{
		NumTokens:        3,
		Dim:              4,
		Depth:            2,
		SeqLen:           2,
		FFMult:           4,
		AttnDim:          0,
		ProbSurvival:     1,
		Causal:           false,
		EmbeddingsDBPath: embeddingsDBPath,
	})
	assert.NotNil(t, model)
	defer model.Close()

	embeddings := map[string][]mat.Float{
		"w0": {0.01, 0.02, 0.03, 0.04},
		"w1": {0.11, 0.12, 0.13, 0.14},
		"w2": {0.21, 0.22, 0.23, 0.24},
	}
	for k, v := range embeddings {
		model.ToEmbed.SetEmbeddingFromData(k, v)
	}

	require.Len(t, model.Layers, 2)
	model.Layers[0].PreNorm.Norm.W.Value().SetData([]mat.Float{0.1, 0.2, 0.3, 0.4})
	model.Layers[0].PreNorm.Norm.B.Value().SetData([]mat.Float{0.5, 0.6, 0.7, 0.8})

	model.Layers[0].PreNorm.Block.Layers[0].(*linear.Model).W.Value().SetData([]mat.Float{
		0.01, 0.02, 0.03, 0.04,
		0.05, 0.06, 0.07, 0.08,
		0.09, 0.10, 0.11, 0.12,
		0.13, 0.14, 0.15, 0.16,
		0.17, 0.18, 0.29, 0.20,
		0.21, 0.22, 0.23, 0.24,
		0.25, 0.26, 0.27, 0.28,
		0.29, 0.30, 0.31, 0.32,
		0.33, 0.34, 0.35, 0.36,
		0.37, 0.38, 0.39, 0.40,
		0.41, 0.42, 0.43, 0.44,
		0.45, 0.46, 0.47, 0.48,
		0.49, 0.50, 0.51, 0.52,
		0.53, 0.54, 0.55, 0.56,
		0.57, 0.58, 0.59, 0.60,
		0.61, 0.62, 0.63, 0.64,
	})
	model.Layers[0].PreNorm.Block.Layers[0].(*linear.Model).B.Value().SetData([]mat.Float{
		0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80,
	})

	model.Layers[0].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Norm.W.Value().SetData([]mat.Float{
		0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7,
	})
	model.Layers[0].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Norm.B.Value().SetData([]mat.Float{
		0.02, 0.04, 0.06, 0.08, 0.01, 0.03, 0.05, 0.07,
	})

	model.Layers[0].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Proj.W.Value().SetData([]mat.Float{
		0.41, 0.42,
		0.43, 0.44,
	})
	model.Layers[0].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Proj.B.Value().SetData([]mat.Float{
		0.48, 0.49,
	})

	model.Layers[0].PreNorm.Block.Layers[3].(*linear.Model).W.Value().SetData([]mat.Float{
		0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88,
		0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99,
		0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.88,
		0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.88, 0.77,
	})
	model.Layers[0].PreNorm.Block.Layers[3].(*linear.Model).B.Value().SetData([]mat.Float{
		0.55, 0.66, 0.77, 0.88,
	})

	// ---

	model.Layers[1].PreNorm.Norm.W.Value().SetData([]mat.Float{0.9, 0.8, 0.7, 0.6})
	model.Layers[1].PreNorm.Norm.B.Value().SetData([]mat.Float{0.5, 0.4, 0.3, 0.2})

	model.Layers[1].PreNorm.Block.Layers[0].(*linear.Model).W.Value().SetData([]mat.Float{
		0.99, 0.98, 0.97, 0.96,
		0.95, 0.94, 0.93, 0.92,
		0.91, 0.90, 0.89, 0.88,
		0.87, 0.86, 0.85, 0.84,
		0.83, 0.82, 0.81, 0.80,
		0.79, 0.78, 0.77, 0.76,
		0.75, 0.74, 0.73, 0.72,
		0.71, 0.70, 0.69, 0.68,
		0.67, 0.66, 0.65, 0.64,
		0.63, 0.62, 0.61, 0.60,
		0.59, 0.58, 0.57, 0.56,
		0.55, 0.54, 0.53, 0.52,
		0.51, 0.50, 0.49, 0.48,
		0.47, 0.46, 0.45, 0.44,
		0.43, 0.42, 0.41, 0.40,
		0.39, 0.38, 0.37, 0.36,
	})
	model.Layers[1].PreNorm.Block.Layers[0].(*linear.Model).B.Value().SetData([]mat.Float{
		0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20,
	})

	model.Layers[1].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Norm.W.Value().SetData([]mat.Float{
		0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2,
	})
	model.Layers[1].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Norm.B.Value().SetData([]mat.Float{
		0.09, 0.07, 0.05, 0.03, 0.08, 0.06, 0.04, 0.02,
	})

	model.Layers[1].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Proj.W.Value().SetData([]mat.Float{
		0.61, 0.62,
		0.63, 0.64,
	})
	model.Layers[1].PreNorm.Block.Layers[2].(*SpatialGatingUnit).Proj.B.Value().SetData([]mat.Float{
		0.68, 0.69,
	})

	model.Layers[1].PreNorm.Block.Layers[3].(*linear.Model).W.Value().SetData([]mat.Float{
		0.99, 0.88, 0.77, 0.66, 0.55, 0.44, 0.33, 0.22,
		0.88, 0.77, 0.66, 0.55, 0.44, 0.33, 0.22, 0.11,
		0.77, 0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0.22,
		0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0.22, 0.33,
	})
	model.Layers[1].PreNorm.Block.Layers[3].(*linear.Model).B.Value().SetData([]mat.Float{
		0.55, 0.44, 0.33, 0.22,
	})

	model.ToLogits.Layers[0].(*layernorm.Model).W.Value().SetData([]mat.Float{0.5, 0.6, 0.7, 0.8})
	model.ToLogits.Layers[0].(*layernorm.Model).B.Value().SetData([]mat.Float{0.05, 0.06, 0.07, 0.08})

	model.ToLogits.Layers[1].(*linear.Model).W.Value().SetData([]mat.Float{
		0.12, 0.34, 0.56, 0.78,
		0.23, 0.45, 0.67, 0.89,
		0.98, 0.76, 0.54, 0.32,
	})
	model.ToLogits.Layers[1].(*linear.Model).B.Value().SetData([]mat.Float{0.19, 0.28, 0.37})

	g := ag.NewGraph()
	defer g.Clear()
	proc := nn.ReifyForTraining(model, g).(*Model)

	words := []string{"w1", "w2"}

	ys := proc.ForwardWords(words)
	require.Len(t, ys, 2)
	require.InDeltaSlice(t, []mat.Float{-0.5176, -0.4466, 0.8876}, ys[0].Value().Data(), 0.00005)
	require.InDeltaSlice(t, []mat.Float{-0.5172, -0.4461, 0.8876}, ys[1].Value().Data(), 0.00005)

	out := g.Add(ys[0], ys[1])
	g.Backward(out)

	emb, ok := model.ToEmbed.UsedEmbeddings.Load("w1")
	assert.True(t, ok)
	assert.NotNil(t, emb)
}

func initializeEmbeddings(model *Model, wordsCount, vecSize int) {
	emb := model.ToEmbed
	for i := 0; i < wordsCount; i++ {
		word := fmt.Sprintf("%d", i)

		vec := make([]mat.Float, vecSize)
		for j := range vec {
			vec[j] = mat.Float(i) / mat.Float(wordsCount)
		}
		emb.SetEmbeddingFromData(word, vec)
	}
}

func createTempDir(t *testing.T, pattern string) string {
	name, err := os.MkdirTemp("", pattern)
	require.NoError(t, err)
	return name
}

func deleteDir(t *testing.T, name string) {
	require.NoError(t, os.RemoveAll(name))
}

// TODO: check release/clear everywhere!
