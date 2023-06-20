// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "github.com/nlpodyssey/spago/mat/float"

// Shape returns the size in each dimension.
func (o *Operator) Shape() []int {
	return o.Value().Shape()
}

// Dims returns the number of dimensions.
func (o *Operator) Dims() int {
	return o.Value().Dims()
}

// Size returns the total number of elements.
func (o *Operator) Size() int {
	return o.Value().Size()
}

// Data returns the underlying data of the tensor.
func (o *Operator) Data() float.Slice {
	return o.Value().Data()
}
