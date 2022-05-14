// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math"
)

// FloatSliceInterface is implemented by any value that can be resolved
// to a slice of a type constrained by DType.
type FloatSliceInterface interface {
	Float32() []float32
	Float64() []float64
	Len() int
	Equals(other FloatSliceInterface) bool
	InDelta(other FloatSliceInterface, delta float64) bool
}

// FloatSlice converts a concrete slice of DType values to an internal
// representation compatible with FloatSliceInterface.
func FloatSlice[T DType](v []T) FloatSliceInterface {
	return floatSlice[T](v)
}

// DTFloatSlice converts a FloatSliceInterface value to a concrete slice
// of DType values.
func DTFloatSlice[T DType](v FloatSliceInterface) []T {
	switch any(T(0)).(type) {
	case float32:
		return any(v.Float32()).([]T)
	case float64:
		return any(v.Float64()).([]T)
	default:
		panic(fmt.Errorf("mat: unexpected slice type []%T", T(0)))
	}
}

// floatSlice is the built-in implementation of a FloatSliceInterface.
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
func (fs floatSlice[T]) Equals(other FloatSliceInterface) bool {
	l := other.Len()
	if len(fs) != l {
		return false
	}
	if l == 0 {
		return true
	}
	o := DTFloatSlice[T](other)
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
func (fs floatSlice[T]) InDelta(other FloatSliceInterface, delta float64) bool {
	l := other.Len()
	if len(fs) != l {
		return false
	}
	if l == 0 {
		return true
	}
	o := DTFloatSlice[T](other)
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
