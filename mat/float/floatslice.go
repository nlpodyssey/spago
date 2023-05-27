// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package float

import (
	"fmt"
	"math"
)

// Slice is implemented by any value that can be resolved
// to a slice of a type constrained by DType.
type Slice interface {
	F32() []float32
	F64() []float64
	BitSize() int
	Len() int
	Equals(other Slice) bool
	InDelta(other Slice, delta float64) bool
}

// Make converts a concrete slice of DType values to an internal
// representation compatible with Slice.
func Make[T DType](v ...T) Slice {
	return floatSlice[T](v)
}

// SliceValueOf converts a Slice value to a concrete slice
// of DType values.
func SliceValueOf[T DType](v Slice) []T {
	switch any(T(0)).(type) {
	case float32:
		return any(v.F32()).([]T)
	case float64:
		return any(v.F64()).([]T)
	default:
		panic(fmt.Errorf("mat: unexpected slice type []%T", T(0)))
	}
}

// floatSlice is the built-in implementation of a Slice.
type floatSlice[T DType] []T

// F32 returns the value as []float32, converting it if necessary.
func (fs floatSlice[T]) F32() []float32 {
	return convertFloatSlice[T, float32](fs)
}

// F64 returns the value as []float64, converting it if necessary.
func (fs floatSlice[T]) F64() []float64 {
	return convertFloatSlice[T, float64](fs)
}

// BitSize returns the size in bits of the internal float value type.
func (fs floatSlice[T]) BitSize() int {
	switch any(T(0)).(type) {
	case float32:
		return 32
	case float64:
		return 64
	default:
		panic(fmt.Errorf("mat: unexpected value type %T", T(0)))
	}
}

// Len returns the length of the slice.
func (fs floatSlice[_]) Len() int {
	return len(fs)
}

// Equals reports whether the content of the receiver is equal to the
// content of the other slice.
// The data type of the other slice is converted to the same type of
// the receiver, if necessary.
func (fs floatSlice[T]) Equals(other Slice) bool {
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
func (fs floatSlice[T]) InDelta(other Slice, delta float64) bool {
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
