// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPositionalEncoder_Forward(t *testing.T) {
	encoder := NewPositionalEncoder(4, 100)

	y1 := encoder.EncodingAt(10)
	y2 := encoder.EncodingAt(0)

	assert.InDeltaSlice(t, []mat.Float{-0.544021, -0.8390715, 0.0998334, 0.99500416}, y1.Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.0, 1.0, 0.0, 1.0}, y2.Data(), 0.000001)
}

func TestAxialPositionalEncoder_Forward(t *testing.T) {
	encoder := NewAxialPositionalEncoder(4, 2, 100, 10, 10)

	y1 := encoder.EncodingAt(15)
	y2 := encoder.EncodingAt(50)
	y3 := encoder.EncodingAt(0)
	y4 := encoder.EncodingAt(99)

	assert.InDeltaSlice(t, []mat.Float{-0.9589242, 0.2836621, 0.0099998, 0.99995000}, y1.Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.0, 1.0, 0.049979169, 0.9987502}, y2.Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.0, 1.0, 0.0, 1.0}, y3.Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.4121184, -0.9111302, 0.0898785, 0.99595273}, y4.Data(), 0.000001)
}
