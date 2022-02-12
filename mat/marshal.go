// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const (
	binaryNilMatrix byte = iota
	binaryDenseMatrix
)

// MarshalBinaryMatrix encodes a Matrix into binary form.
func MarshalBinaryMatrix[T DType](m Matrix[T], w io.Writer) error {
	var mType byte
	var bin []byte
	var err error

	switch v := m.(type) {
	case nil:
		mType = binaryNilMatrix
	case *Dense[T]:
		mType = binaryDenseMatrix
		bin, err = v.MarshalBinary()
	default:
		return fmt.Errorf("unknown matrix type %T: %#v", m, m)
	}

	if err != nil {
		return err
	}

	_, err = w.Write([]byte{mType})
	if err != nil {
		return err
	}
	if mType == binaryNilMatrix {
		return nil
	}

	binLen := make([]byte, 4)
	binary.LittleEndian.PutUint32(binLen, uint32(len(bin)))
	_, err = w.Write(binLen)
	if err != nil {
		return err
	}

	_, err = w.Write(bin)
	if err != nil {
		return err
	}

	return nil
}

// UnmarshalBinaryMatrix decodes a Matrix from binary form.
func UnmarshalBinaryMatrix[T DType](r io.Reader) (Matrix[T], error) {
	smType := make([]byte, 1)
	_, err := r.Read(smType)
	if err != nil {
		return nil, err
	}
	mType := smType[0]
	if mType == binaryNilMatrix {
		return nil, nil
	}

	binLenBytes := make([]byte, 4)
	_, err = r.Read(binLenBytes)
	if err != nil {
		return nil, err
	}
	binLen := int(binary.LittleEndian.Uint32(binLenBytes))
	bin := make([]byte, binLen)
	_, err = r.Read(bin)
	if err != nil {
		return nil, err
	}

	switch mType {
	case binaryDenseMatrix:
		m := new(Dense[T])
		err = m.UnmarshalBinary(bin)
		return m, err
	default:
		return nil, fmt.Errorf("unknown binary matrix type %d", mType)
	}
}

// MarshalBinary marshals a Dense matrix into binary form.
func (d Dense[T]) MarshalBinary() ([]byte, error) {
	switch any(T(0)).(type) {
	case float32:
		data := make([]byte, 8+d.Size()*4)
		binary.LittleEndian.PutUint32(data, uint32(d.rows))
		binary.LittleEndian.PutUint32(data[4:], uint32(d.cols))
		for i, v := range d.data {
			binary.LittleEndian.PutUint32(data[8+i*4:], math.Float32bits(float32(v)))
		}
		return data, nil
	case float64:
		data := make([]byte, 8+d.Size()*8)
		binary.LittleEndian.PutUint32(data, uint32(d.rows))
		binary.LittleEndian.PutUint32(data[4:], uint32(d.cols))
		for i, v := range d.data {
			binary.LittleEndian.PutUint64(data[8+i*8:], math.Float64bits(float64(v)))
		}
		return data, nil
	default:
		panic(fmt.Sprintf("mat: unexpected dense matrix type %T", T(0)))
	}
}

// UnmarshalBinary unmarshals a binary representation of a Dense matrix.
func (d *Dense[T]) UnmarshalBinary(data []byte) error {
	d.flags = 0
	d.rows = int(binary.LittleEndian.Uint32(data))
	d.cols = int(binary.LittleEndian.Uint32(data[4:]))
	size := d.rows * d.cols
	d.data = make([]T, size)

	switch any(T(0)).(type) {
	case float32:
		if expected := 8 + size*4; len(data) != expected {
			return fmt.Errorf("mat: cannot unmarshal float32 Dense matrix. Data len expected %d, actual %d", expected, len(data))
		}
		for i := range d.data {
			d.data[i] = T(math.Float32frombits(binary.LittleEndian.Uint32(data[8+i*4:])))
		}
	case float64:
		if expected := 8 + size*8; len(data) != expected {
			return fmt.Errorf("mat: cannot unmarshal float64 Dense matrix. Data len expected %d, actual %d", expected, len(data))
		}
		for i := range d.data {
			d.data[i] = T(math.Float64frombits(binary.LittleEndian.Uint64(data[8+i*8:])))
		}
	default:
		panic(fmt.Sprintf("mat: unexpected dense matrix type %T", T(0)))
	}

	return nil
}
