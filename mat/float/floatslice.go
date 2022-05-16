// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package float

import (
	"fmt"
	"math"
)

// SliceInterface is implemented by any value that can be resolved
// to a slice of a type constrained by DType.
type SliceInterface interface {
	Float32() []float32
	Float64() []float64
	Len() int
	Equals(other SliceInterface) bool
	InDelta(other SliceInterface, delta float64) bool
}

// Slice converts a concrete slice of DType values to an internal
// representation compatible with SliceInterface.
func Slice[T DType](v []T) SliceInterface {
	return floatSlice[T](v)
}

// SliceValueOf converts a SliceInterface value to a concrete slice
// of DType values.
func SliceValueOf[T DType](v SliceInterface) []T {
	switch any(T(0)).(type) {
	case float32:
		return any(v.Float32()).([]T)
	case float64:
		return any(v.Float64()).([]T)
	default:
		panic(fmt.Errorf("mat: unexpected slice type []%T", T(0)))
	}
}

// floatSlice is the built-in implementation of a SliceInterface.
type floatSlice[T DType] []T

// Float32 returns the value as []float32, converting it if necessary.
func (fs floatSlice[T]) Float32() []float32 {
	return convertFloatSlice[T, float32](fs)
}

// Float64 returns the value as []float64, converting it if necessary.
func (fs floatSlice[T]) Float64() []float64 {
	return convertFloatSlice[T, float64](fs)
}

// Len returns the length of the slice.
func (fs floatSlice[_]) Len() int {
	return len(fs)
}

// Equals reports whether the content of the receiver is equal to the
// content of the other slice.
// The data type of the other slice is converted to the same type of
// the receiver, if necessary.
func (fs floatSlice[T]) Equals(other SliceInterface) bool {
	l := other.Len()
	if len(fs) != l {
		return false
	}
	if l == 0 {
		return true
	}
	o := SliceValueOf[T](other)
	_ = o[len(fs)-1]
	for i, v := range fs {
		if v != o[i] {
			return false
		}
	}
	return true
}

// InDelta reports whether the receiver and the other slice have the same
// length and all their values at the same positions are within delta.
// The data type of the other slice is converted to the same type of
// the receiver, if necessary.
func (fs floatSlice[T]) InDelta(other SliceInterface, delta float64) bool {
	l := other.Len()
	if len(fs) != l {
		return false
	}
	if l == 0 {
		return true
	}
	o := SliceValueOf[T](other)
	_ = o[len(fs)-1]
	for i, v := range fs {
		if math.Abs(float64(v-o[i])) > delta {
			return false
		}
	}
	return true
}

// convertFloatSlice converts the given source slice to the destination type
// if necessary. If the source is nil, it returns nil. If the source type
// is identical to the destination type, the source is returned directly.
func convertFloatSlice[S, D DType](source []S) []D {
	if v, ok := any(source).([]D); ok {
		return v
	}
	if source == nil {
		return nil
	}
	dest := make([]D, len(source))
	for i, v := range source {
		dest[i] = D(v)
	}
	return dest
}
