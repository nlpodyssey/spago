// Copyright 2023 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/mat/float"
)

// Tensor represents an interface for a generic tensor.
type Tensor interface {
	// Shape returns the size in each dimension.
	Shape() []int
	// Dims returns the number of dimensions.
	Dims() int
	// Size returns the total number of elements.
	Size() int
	// Data returns the underlying data of the tensor.
	Data() float.Slice
	// Item returns the scalar value.
	// It panics if the matrix does not contain exactly one element.
	Item() float.Float
	// SetAt sets the value at the given indices.
	// It panics if the given indices are out of range.
	SetAt(m Tensor, indices ...int)
	// At returns the value at the given indices.
	// It panics if the given indices are out of range.
	At(indices ...int) Tensor
	// Value returns the value of the node.
	// In case of a leaf node, it returns the value of the underlying matrix.
	// In case of a non-leaf node, it returns the value of the operation performed during the forward pass.
	Value() Tensor
	// Grad returns the gradients accumulated during the backward pass.
	// A matrix full of zeros and the nil value are considered equivalent.
	Grad() Tensor
	// HasGrad reports whether there are accumulated gradients.
	HasGrad() bool
	// RequiresGrad reports whether the node requires gradients.
	RequiresGrad() bool
	// AccGrad accumulates the gradients into the node.
	AccGrad(gx Tensor)
	// ZeroGrad zeroes the gradients, setting the value of Grad to nil.
	ZeroGrad()
}

func init() {
	gob.Register([]Tensor{})
}
