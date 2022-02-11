// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewSinusoidalPositionalEncoder(t *testing.T) {
	t.Run("even size and length", func(t *testing.T) {
		enc := NewSinusoidalPositionalEncoder[mat.Float](6, 4)
		assert.Equal(t, 4, len(enc.Vectors))
		assert.InDeltaSlice(t, []mat.Float{0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000}, enc.Vectors[0].Data(), 0.0001)
		assert.InDeltaSlice(t, []mat.Float{0.8415, 0.0464, 0.0022, 0.5403, 0.9989, 1.0000}, enc.Vectors[1].Data(), 0.0001)
		assert.InDeltaSlice(t, []mat.Float{0.9093, 0.0927, 0.0043, -0.4161, 0.9957, 1.0000}, enc.Vectors[2].Data(), 0.0001)
		assert.InDeltaSlice(t, []mat.Float{0.1411, 0.1388, 0.0065, -0.9900, 0.9903, 1.0000}, enc.Vectors[3].Data(), 0.0001)
	})

	t.Run("odd size and length", func(t *testing.T) {
		enc := NewSinusoidalPositionalEncoder[mat.Float](5, 3)
		assert.Equal(t, 3, len(enc.Vectors))
		assert.InDeltaSlice(t, []mat.Float{0, 0, 0, 1, 1}, enc.Vectors[0].Data(), 0.00001)
		assert.InDeltaSlice(t, []mat.Float{0.8415, 0.0251, 0.0006, 0.5403, 0.9997}, enc.Vectors[1].Data(), 0.0001)
		assert.InDeltaSlice(t, []mat.Float{0.9093, 0.0502, 0.00126, -0.4162, 0.9987}, enc.Vectors[2].Data(), 0.0001)
	})
}
