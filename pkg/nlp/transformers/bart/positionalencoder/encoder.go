// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package positionalencoder

import "github.com/nlpodyssey/spago/pkg/ml/ag"

// Encoder is implemented by Models that require positional encoding.
type Encoder interface {
	// Encode transforms the input positions into a dense representation.
	Encode(positions []int) []ag.Node
}
