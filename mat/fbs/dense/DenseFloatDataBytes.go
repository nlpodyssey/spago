package dense

import flatbuffers "github.com/google/flatbuffers/go"

const (
	DTypeFloat32 int32 = 0
	DTypeFloat64 int32 = 1
)

func (rcv *DenseFloat32) DataBytes() []byte {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(10))
	if o != 0 {
		a := rcv._tab.Vector(o)
		return rcv._tab.Bytes[a : int(a)+int(rcv._tab.VectorLen(o))*4]
	}
	return nil
}

func (rcv *DenseFloat64) DataBytes() []byte {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(10))
	if o != 0 {
		a := rcv._tab.Vector(o)
		return rcv._tab.Bytes[a : int(a)+int(rcv._tab.VectorLen(o))*8]
	}
	return nil
}
