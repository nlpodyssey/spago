// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gopickleutils

import (
	"github.com/nlpodyssey/gopickle/pytorch"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

// GetData returns the data of a PyTorch tensor as a mat.Float slice.
// It returns the data using the row-major representation, possibly converting column-major order to row-major order.
func GetData(t *pytorch.Tensor) []mat.Float {
	if len(t.Size) == 0 || len(t.Size) > 2 {
		panic("gopickleutils: number of sizes not supported")
	}
	size := t.Size[0]
	if len(t.Size) > 1 {
		size *= t.Size[1]
	}
	orig := t.Source.(*pytorch.FloatStorage).Data[t.StorageOffset : t.StorageOffset+size]
	data := make([]mat.Float, len(orig))

	if len(t.Size) == 1 || t.Size[0] == 1 || t.Size[1] == 1 || t.Stride[1] == 1 {
		for i, val := range orig {
			data[i] = mat.Float(val)
		}
		return data
	}

	s0, s1 := t.Size[1], t.Size[0]
	for i := 0; i < s0; i++ {
		for j := 0; j < s1; j++ {
			data[i+j*s0] = mat.Float(orig[j+i*s1])
		}
	}
	return data
}
