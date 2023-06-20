// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// Buffer is a type of Node that do not require gradients but that can be serialized similarly
// to any other parameters.
// This is useful e.g. to store constants, to track the mean and std in batch norm layers etc.
type Buffer struct {
	mat.Tensor
}

func init() {
	gob.Register(&Buffer{})
}

// Buf creates a new Buffer Node.
func Buf(value mat.Tensor) *Buffer {
	return &Buffer{
		Tensor: ag.StopGrad(value),
	}
}
