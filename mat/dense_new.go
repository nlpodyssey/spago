// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/mat/float"
)

type Options struct {
	RequiresGrad bool // default: false
	Shape        []int
	Slice        float.Slice
}

type OptionsFunc func(opt *Options)

func WithGrad(value bool) OptionsFunc {
	return func(opts *Options) {
		opts.RequiresGrad = value
	}
}

func WithShape(shape ...int) OptionsFunc {
	return func(opts *Options) {
		opts.Shape = shape
	}
}

func WithBacking[T float.DType](data []T) OptionsFunc {
	return func(opts *Options) {
		opts.Slice = float.Make[T](data...)
	}
}

func NewDense[T float.DType](opts ...OptionsFunc) *Dense[T] {
	r, err := newDense[T](opts...)
	if err != nil {
		panic(err) // TODO: do not panic
	}
	return r
}

func Scalar[T float.DType](value T, opts ...OptionsFunc) *Dense[T] {
	r, err := newScalar[T](value, opts...)
	if err != nil {
		panic(err) // TODO: do not panic
	}
	return r
}

func newDense[T float.DType](opts ...OptionsFunc) (*Dense[T], error) {
	args := &Options{}
	for _, opt := range opts {
		opt(args)
	}

	if args.Slice != nil {
		return newDenseFromSlice[T](args)
	} else if args.Shape != nil {
		return newDenseFromShape[T](args)
	}

	return nil, fmt.Errorf("mat: shape or slice must be specified")
}

func newDenseFromSlice[T float.DType](args *Options) (*Dense[T], error) {
	if args.Shape == nil {
		args.Shape = []int{args.Slice.Len(), 1}
	}

	if err := checkShape(args.Shape...); err != nil {
		return nil, err
	}
	shape := adjustShape(args.Shape...)
	size := calculateSize(shape)

	if args.Slice.Len() != size {
		return nil, fmt.Errorf("mat: wrong dimensions. Expected %d, got %d", size, args.Slice.Len())
	}

	return &Dense[T]{
		shape:        shape,
		data:         float.SliceValueOf[T](args.Slice),
		requiresGrad: args.RequiresGrad,
	}, nil
}

func newDenseFromShape[T float.DType](args *Options) (*Dense[T], error) {
	if err := checkShape(args.Shape...); err != nil {
		return nil, err
	}
	shape := adjustShape(args.Shape...)
	size := calculateSize(shape)

	return &Dense[T]{
		shape:        shape,
		data:         make([]T, size),
		requiresGrad: args.RequiresGrad,
	}, nil
}

func calculateSize(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

func checkShape(shape ...int) error {
	if len(shape) < 1 || len(shape) > 2 {
		return fmt.Errorf("mat: wrong matrix dimensions. Must be 1 or 2")
	}
	for _, s := range shape {
		if s < 0 {
			return fmt.Errorf("mat: negative value for shape is not allowed")
		}
	}
	return nil
}

func adjustShape(shape ...int) []int {
	if len(shape) == 1 {
		return []int{shape[0], 1}
	}
	return shape
}

func newScalar[T float.DType](value T, opts ...OptionsFunc) (*Dense[T], error) {
	args := &Options{}
	for _, opt := range opts {
		opt(args)
	}
	if args.Shape != nil || args.Slice != nil {
		log.Fatal("mat: WithShape and WithBacking options are not allowed when creating a scalar")
	}
	scalarOpts := append(opts, WithBacking([]T{value}))
	return newDense[T](scalarOpts...)
}

func CreateInitializedSlice[T float.DType](size int, v T) []T {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := malloc[T](size)
	for i := range out {
		out[i] = v
	}
	return out
}

func CreateOneHotVector[T float.DType](size int, index int) []T {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := malloc[T](size)
	out[index] = 1
	return out
}

func CreateIdentityMatrix[T float.DType](size int) []T {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := malloc[T](size * size)
	for i := 0; i < len(out); i += size + 1 {
		out[i] = 1
	}
	return out
}

func InitializeMatrix[T float.DType](rows, cols int, fn func(r, c int) T) []T {
	if rows < 0 || cols < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := malloc[T](rows * cols)
	r := 0
	c := 0
	for i := range out {
		out[i] = fn(r, c)
		c++
		if c == cols {
			r++
			c = 0
		}
	}
	return out
}
