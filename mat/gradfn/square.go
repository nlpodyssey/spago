// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import "github.com/nlpodyssey/spago/mat"

// Square is an operator to perform element-wise square.
type Square[O mat.Tensor] struct {
	*Prod[O]
}

// NewSquare returns a new Square Function.
func NewSquare[O mat.Tensor](x O) *Square[O] {
	return &Square[O]{
		Prod: &Prod[O]{x1: x, x2: x},
	}
}
