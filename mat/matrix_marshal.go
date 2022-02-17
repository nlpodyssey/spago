// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"encoding/binary"
	"fmt"
	"io"
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
