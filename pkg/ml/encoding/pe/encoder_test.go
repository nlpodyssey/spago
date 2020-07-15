// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestPositionalEncoder_Forward(t *testing.T) {
	encoder := NewPositionalEncoder(4, 100)

	y1 := encoder.EncodingAt(10)
	y2 := encoder.EncodingAt(0)

	if !floats.EqualApprox(y1.Data(), []float64{-0.544021, -0.8390715, 0.0998334, 0.99500416}, 0.000001) {
		t.Error("First position doesn't match the expected values")
	}
	if !floats.EqualApprox(y2.Data(), []float64{0.0, 1.0, 0.0, 1.0}, 0.000001) {
		t.Error("Second position doesn't match the expected values")
	}
}

func TestAxialPositionalEncoder_Forward(t *testing.T) {
	encoder := NewAxialPositionalEncoder(4, 2, 100, 10, 10)

	y1 := encoder.EncodingAt(15)
	y2 := encoder.EncodingAt(50)
	y3 := encoder.EncodingAt(0)
	y4 := encoder.EncodingAt(99)

	if !floats.EqualApprox(y1.Data(), []float64{-0.9589242, 0.2836621, 0.0099998, 0.99995000}, 0.000001) {
		t.Error("First position doesn't match the expected values")
	}
	if !floats.EqualApprox(y2.Data(), []float64{0.0, 1.0, 0.049979169, 0.9987502}, 0.000001) {
		t.Error("Second position doesn't match the expected values")
	}
	if !floats.EqualApprox(y3.Data(), []float64{0.0, 1.0, 0.0, 1.0}, 0.000001) {
		t.Error("Third position doesn't match the expected values")
	}
	if !floats.EqualApprox(y4.Data(), []float64{0.4121184, -0.9111302, 0.0898785, 0.99595273}, 0.000001) {
		t.Error("Fourth position doesn't match the expected values")
	}
}
