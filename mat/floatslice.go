// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import "fmt"

// FloatSliceInterface is implemented by any value that can be resolved
// to a slice of a type constrained by DType.
type FloatSliceInterface interface {
	Float32() []float32
	Float64() []float64
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
