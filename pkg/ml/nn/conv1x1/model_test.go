// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conv1x1

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	t.Run("input 3, output 2", func(t *testing.T) {
		model := New[mat.Float](Config{
			InputChannels:  3,
			OutputChannels: 2,
		})
		defer model.Close()

		require.Equal(t, 2, model.W.Value().Rows())
		require.Equal(t, 3, model.W.Value().Columns())
		require.Equal(t, 2, model.B.Value().Rows())
		require.Equal(t, 1, model.B.Value().Columns())

		model.B.Value().SetData([]mat.Float{0.1, 0.2})
		model.W.Value().SetData([]mat.Float{
			1, 2, 3,
			4, 5, 6,
		})

		g := ag.NewGraph[mat.Float]()
		defer g.Clear()

		proc := nn.ReifyForInference(model, g)

		xs := []ag.Node[mat.Float]{
			g.NewVariable(mat.NewVecDense([]mat.Float{1, 2, 4, 0, -1}), false),
			g.NewVariable(mat.NewVecDense([]mat.Float{1, 3, 3, 0, -1}), false),
			g.NewVariable(mat.NewVecDense([]mat.Float{1, 4, 2, 0, -1}), false),
		}
		ys := proc.Forward(xs...)
		require.Len(t, ys, 2)
		require.True(t, mat.IsVector(ys[0].Value()))
		require.Equal(t, 5, ys[0].Value().Size())
		require.True(t, mat.IsVector(ys[1].Value()))
		require.Equal(t, 5, ys[1].Value().Size())
		assert.InDeltaSlice(t, []mat.Float{6.1, 20.1, 16.1, 0.1, -5.9}, ys[0].Value().Data(), 0.001)
		assert.InDeltaSlice(t, []mat.Float{15.2, 47.2, 43.2, 0.2, -14.8}, ys[1].Value().Data(), 0.001)
	})

	t.Run("input 4, output 3", func(t *testing.T) {
		model := New[mat.Float](Config{
			InputChannels:  4,
			OutputChannels: 3,
		})
		defer model.Close()

		model.B.Value().SetData([]mat.Float{0.6, 0.5, 0.7})
		model.W.Value().SetData([]mat.Float{
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 0.8, 0.7, 0.6,
		})

		g := ag.NewGraph[mat.Float]()
		defer g.Clear()

		proc := nn.ReifyForInference(model, g)

		xs := []ag.Node[mat.Float]{
			g.NewVariable(mat.NewVecDense([]mat.Float{0.2, 0.9, 0.1}), false),
			g.NewVariable(mat.NewVecDense([]mat.Float{0.4, 0.7, 0.1}), false),
			g.NewVariable(mat.NewVecDense([]mat.Float{0.6, 0.5, 0.1}), false),
			g.NewVariable(mat.NewVecDense([]mat.Float{0.8, 0.3, 0.1}), false),
		}
		ys := proc.Forward(xs...)
		assert.InDeltaSlice(t, []mat.Float{1.2, 1.1, 0.7}, ys[0].Value().Data(), 0.001)
		assert.InDeltaSlice(t, []mat.Float{1.9, 1.96, 0.76}, ys[1].Value().Data(), 0.001)
		assert.InDeltaSlice(t, []mat.Float{2.1, 2.6, 1}, ys[2].Value().Data(), 0.001)
	})
}
