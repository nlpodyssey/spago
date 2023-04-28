// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
)

func init() {
	gob.Register(&Dense[float32]{})
	gob.Register(&Dense[float64]{})
}

const (
	binaryDenseFloat32 byte = iota
	binaryDenseFloat64
)

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

// Dense matrix - float32 marshaling:
// - 1 byte - identifier binaryDenseFloat32 (byte)
// - 1 byte - requiresGrad (bool)
// - 8 bytes - rows (uint64)
// - 8 bytes - cols (uint64)
// - 4*size bytes - data (float32 as uint32-bits)

func (d *Dense[T]) marshalBinaryFloat32() ([]byte, error) {
	data := make([]byte, 18+len(d.data)*4)
	data[0] = binaryDenseFloat32
	data[1] = boolToByte(d.requiresGrad)
	binary.LittleEndian.PutUint64(data[2:], uint64(d.rows))
	binary.LittleEndian.PutUint64(data[10:], uint64(d.cols))

	s := data[18:]
	for _, v := range d.data {
		binary.LittleEndian.PutUint32(s, math.Float32bits(float32(v)))
		s = s[4:]
	}

	return data, nil
}

func boolToByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}

func byteToBool(b byte) bool {
	return b == 1
}

func (d *Dense[T]) unmarshalBinaryFloat32(data []byte) error {
	if data[0] != binaryDenseFloat32 {
		return errors.New("mat: cannot unmarshal Dense[float32]: invalid identifier")
	}
	d.requiresGrad = byteToBool(data[1])
	d.rows = int(binary.LittleEndian.Uint64(data[2:]))
	d.cols = int(binary.LittleEndian.Uint64(data[10:]))
	size := d.rows * d.cols
	d.data = make([]T, size)

	data = data[18:]
	dData := d.data
	for i := range dData {
		dData[i] = T(math.Float32frombits(binary.LittleEndian.Uint32(data)))
		data = data[4:]
	}

	d.isNew = false
	return nil
}

// Dense matrix - float64 marshaling:
// - 1 byte - identifier binaryDenseFloat64 (byte)
// - 1 byte - requiresGrad (bool)
// - 8 bytes - rows (uint64)
// - 8 bytes - cols (uint64)
// - 8*size bytes - data (float64 as uint64-bits)

func (d *Dense[T]) marshalBinaryFloat64() ([]byte, error) {
	data := make([]byte, 18+len(d.data)*8)
	data[0] = binaryDenseFloat64
	data[1] = boolToByte(d.requiresGrad)
	binary.LittleEndian.PutUint64(data[2:], uint64(d.rows))
	binary.LittleEndian.PutUint64(data[10:], uint64(d.cols))

	s := data[18:]
	for _, v := range d.data {
		binary.LittleEndian.PutUint64(s, math.Float64bits(float64(v)))
		s = s[8:]
	}

	return data, nil
}

func (d *Dense[T]) unmarshalBinaryFloat64(data []byte) error {
	if data[0] != binaryDenseFloat64 {
		return errors.New("mat: cannot unmarshal Dense[float64]: invalid identifier")
	}
	d.requiresGrad = byteToBool(data[1])
	d.rows = int(binary.LittleEndian.Uint64(data[2:]))
	d.cols = int(binary.LittleEndian.Uint64(data[10:]))
	size := d.rows * d.cols
	d.data = make([]T, size)

	data = data[18:]
	dData := d.data
	for i := range dData {
		dData[i] = T(math.Float64frombits(binary.LittleEndian.Uint64(data)))
		data = data[8:]
	}

	d.isNew = false

	return nil
}
