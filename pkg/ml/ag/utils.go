// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/pkg/mat"

// Map returns a transformed version of xs with all its components modified according to the mapping function.
// It is useful for applying an operator to a sequence of nodes. Keep in mind that using this function has an overhead
// because of the callback, however insignificant compared to mathematical computations.
func Map[T mat.DType](mapping func(Node[T]) Node[T], xs []Node[T]) []Node[T] {
	ys := make([]Node[T], len(xs))
	for i, x := range xs {
		ys[i] = mapping(x)
	}
	return ys
}
