// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package float

import (
	"fmt"
)

// Interface is implemented by any value that can be resolved
// to the types constrained by DType.
type Interface interface {
	Float32() float32
	Float64() float64
}

// Float converts a concrete DType value to an internal representation
// compatible with Interface.
func Float[T DType](v T) Interface {
	return float[T]{v: v}
}

// ValueOf converts a Interface value to a concrete DType.
func ValueOf[T DType](i Interface) T {
	switch any(T(0)).(type) {
	case float32:
		return T(i.Float32())
	case float64:
		return T(i.Float64())
	default:
		panic(fmt.Errorf("mat: unexpected value type %T", T(0)))
	}
}

// float is the built-in implementation of a Interface.
type float[T DType] struct {
	v T
}

// Float32 returns the value as float32, converting it if necessary.
func (f float[_]) Float32() float32 {
	return float32(f.v)
}

// Float64 returns the value as float64, converting it if necessary.
func (f float[_]) Float64() float64 {
	return float64(f.v)
}

// Float64 returns the value as float64, converting it if necessary.
func (f float[_]) String() string {
	return fmt.Sprint(f.v)
}
