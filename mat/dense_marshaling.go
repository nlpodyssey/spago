// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"encoding/gob"
	"fmt"
	"unsafe"

	flatbuffers "github.com/google/flatbuffers/go"
	"github.com/nlpodyssey/spago/mat/fbs/dense"
)

func init() {
	gob.Register(&Dense[float32]{})
	gob.Register(&Dense[float64]{})
}

// MarshalBinary marshals a Dense matrix into binary form.
func (d *Dense[T]) MarshalBinary() ([]byte, error) {
	switch any(T(0)).(type) {
	case float32:
		return d.marshalBinaryFloat32()
	case float64:
		return d.marshalBinaryFloat64()
	default:
		panic(fmt.Sprintf("mat: unexpected dense matrix type %T", T(0)))
	}
}

// UnmarshalBinary unmarshals a binary representation of a Dense matrix.
func (d *Dense[T]) UnmarshalBinary(data []byte) error {
	switch any(T(0)).(type) {
	case float32:
		return d.unmarshalBinaryFloat32(data)
	case float64:
		return d.unmarshalBinaryFloat64(data)
	default:
		panic(fmt.Sprintf("mat: unexpected dense matrix type %T", T(0)))
	}
}

func (d *Dense[T]) marshalBinaryFloat32() ([]byte, error) {
	b := flatbuffers.NewBuilder(0)

	dense.DenseFloat32StartShapeVector(b, len(d.shape))
	for i := len(d.shape) - 1; i >= 0; i-- {
		b.PrependInt32(int32(d.shape[i]))
	}
	shape := b.EndVector(len(d.shape))

	dense.DenseFloat32StartDataVector(b, len(d.data))
	for i := len(d.data) - 1; i >= 0; i-- {
		b.PrependFloat32(float32(d.data[i]))
	}
	data := b.EndVector(len(d.data))

	dense.DenseFloat32Start(b)
	dense.DenseFloat32AddDtype(b, dense.DTypeFloat32)
	dense.DenseFloat32AddRequiresFrad(b, d.requiresGrad)
	dense.DenseFloat32AddShape(b, shape)
	dense.DenseFloat32AddData(b, data)
	b.Finish(dense.DenseFloat32End(b))

	return b.FinishedBytes(), nil
}

func (d *Dense[T]) unmarshalBinaryFloat32(data []byte) error {
	raw := dense.GetRootAsDenseFloat32(data, 0)

	if raw.Dtype() != dense.DTypeFloat32 {
		return fmt.Errorf("mat: unexpected dtype %v", raw.Dtype())
	}

	d.requiresGrad = raw.RequiresFrad()

	d.shape = make([]int, raw.ShapeLength())
	for i := 0; i < raw.ShapeLength(); i++ {
		d.shape[i] = int(raw.Shape(i))
	}

	d.data = bytesToSlice[T](raw.DataBytes(), raw.DataLength())
	return nil
}

func (d *Dense[T]) marshalBinaryFloat64() ([]byte, error) {
	b := flatbuffers.NewBuilder(0)

	dense.DenseFloat64StartShapeVector(b, len(d.shape))
	for i := len(d.shape) - 1; i >= 0; i-- {
		b.PrependInt32(int32(d.shape[i]))
	}
	shape := b.EndVector(len(d.shape))

	dense.DenseFloat64StartDataVector(b, len(d.data))
	for i := len(d.data) - 1; i >= 0; i-- {
		b.PrependFloat64(float64(d.data[i]))
	}
	data := b.EndVector(len(d.data))

	dense.DenseFloat64Start(b)
	dense.DenseFloat64AddDtype(b, dense.DTypeFloat64)
	dense.DenseFloat64AddRequiresGrad(b, d.requiresGrad)
	dense.DenseFloat64AddShape(b, shape)
	dense.DenseFloat64AddData(b, data)
	b.Finish(dense.DenseFloat64End(b))

	return b.FinishedBytes(), nil
}

func (d *Dense[T]) unmarshalBinaryFloat64(data []byte) error {
	raw := dense.GetRootAsDenseFloat64(data, 0)

	if raw.Dtype() != dense.DTypeFloat64 {
		return fmt.Errorf("mat: unexpected dtype %v", raw.Dtype())
	}

	d.requiresGrad = raw.RequiresGrad()

	d.shape = make([]int, raw.ShapeLength())
	for i := 0; i < raw.ShapeLength(); i++ {
		d.shape[i] = int(raw.Shape(i))
	}

	d.data = bytesToSlice[T](raw.DataBytes(), raw.DataLength())
	return nil
}

func bytesToSlice[T any](b []byte, length int) []T {
	if len(b) == 0 {
		return []T{}
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&b[0])), length)
}
