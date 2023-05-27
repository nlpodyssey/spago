// Code generated by command: go run mulconst_asm.go -out ../../matfuncs/mulconst_amd64.s -stubs ../../matfuncs/mulconst_amd64_stubs.go -pkg matfuncs. DO NOT EDIT.

//go:build amd64 && gc && !purego

#include "textflag.h"

// func MulConstAVX32(c float32, x []float32, y []float32)
// Requires: AVX, AVX2, SSE
TEXT ·MulConstAVX32(SB), NOSPLIT, $0-56
	MOVSS        c+0(FP), X0
	MOVQ         x_base+8(FP), AX
	MOVQ         y_base+32(FP), CX
	MOVQ         x_len+16(FP), DX
	VBROADCASTSS X0, Y1

unrolledLoop14:
	CMPQ    DX, $0x00000070
	JL      unrolledLoop8
	VMULPS  (AX), Y1, Y2
	VMULPS  32(AX), Y1, Y3
	VMULPS  64(AX), Y1, Y4
	VMULPS  96(AX), Y1, Y5
	VMULPS  128(AX), Y1, Y6
	VMULPS  160(AX), Y1, Y7
	VMULPS  192(AX), Y1, Y8
	VMULPS  224(AX), Y1, Y9
	VMULPS  256(AX), Y1, Y10
	VMULPS  288(AX), Y1, Y11
	VMULPS  320(AX), Y1, Y12
	VMULPS  352(AX), Y1, Y13
	VMULPS  384(AX), Y1, Y14
	VMULPS  416(AX), Y1, Y15
	VMOVUPS Y2, (CX)
	VMOVUPS Y3, 32(CX)
	VMOVUPS Y4, 64(CX)
	VMOVUPS Y5, 96(CX)
	VMOVUPS Y6, 128(CX)
	VMOVUPS Y7, 160(CX)
	VMOVUPS Y8, 192(CX)
	VMOVUPS Y9, 224(CX)
	VMOVUPS Y10, 256(CX)
	VMOVUPS Y11, 288(CX)
	VMOVUPS Y12, 320(CX)
	VMOVUPS Y13, 352(CX)
	VMOVUPS Y14, 384(CX)
	VMOVUPS Y15, 416(CX)
	ADDQ    $0x000001c0, AX
	ADDQ    $0x000001c0, CX
	SUBQ    $0x00000070, DX
	JMP     unrolledLoop14

unrolledLoop8:
	CMPQ    DX, $0x00000040
	JL      unrolledLoop4
	VMULPS  (AX), Y1, Y2
	VMULPS  32(AX), Y1, Y3
	VMULPS  64(AX), Y1, Y4
	VMULPS  96(AX), Y1, Y5
	VMULPS  128(AX), Y1, Y6
	VMULPS  160(AX), Y1, Y7
	VMULPS  192(AX), Y1, Y8
	VMULPS  224(AX), Y1, Y9
	VMOVUPS Y2, (CX)
	VMOVUPS Y3, 32(CX)
	VMOVUPS Y4, 64(CX)
	VMOVUPS Y5, 96(CX)
	VMOVUPS Y6, 128(CX)
	VMOVUPS Y7, 160(CX)
	VMOVUPS Y8, 192(CX)
	VMOVUPS Y9, 224(CX)
	ADDQ    $0x00000100, AX
	ADDQ    $0x00000100, CX
	SUBQ    $0x00000040, DX
	JMP     unrolledLoop8

unrolledLoop4:
	CMPQ    DX, $0x00000020
	JL      unrolledLoop1
	VMULPS  (AX), Y1, Y2
	VMULPS  32(AX), Y1, Y3
	VMULPS  64(AX), Y1, Y4
	VMULPS  96(AX), Y1, Y5
	VMOVUPS Y2, (CX)
	VMOVUPS Y3, 32(CX)
	VMOVUPS Y4, 64(CX)
	VMOVUPS Y5, 96(CX)
	ADDQ    $0x00000080, AX
	ADDQ    $0x00000080, CX
	SUBQ    $0x00000020, DX
	JMP     unrolledLoop4

unrolledLoop1:
	CMPQ    DX, $0x00000008
	JL      tailLoop
	VMULPS  (AX), Y1, Y2
	VMOVUPS Y2, (CX)
	ADDQ    $0x00000020, AX
	ADDQ    $0x00000020, CX
	SUBQ    $0x00000008, DX
	JMP     unrolledLoop1

tailLoop:
	CMPQ  DX, $0x00000000
	JE    end
	MOVSS (AX), X1
	MULSS X0, X1
	MOVSS X1, (CX)
	ADDQ  $0x00000004, AX
	ADDQ  $0x00000004, CX
	DECQ  DX
	JMP   tailLoop

end:
	RET

// func MulConstAVX64(c float64, x []float64, y []float64)
// Requires: AVX, AVX2, SSE2
TEXT ·MulConstAVX64(SB), NOSPLIT, $0-56
	MOVSD        c+0(FP), X0
	MOVQ         x_base+8(FP), AX
	MOVQ         y_base+32(FP), CX
	MOVQ         x_len+16(FP), DX
	VBROADCASTSD X0, Y1

unrolledLoop14:
	CMPQ    DX, $0x00000038
	JL      unrolledLoop8
	VMULPD  (AX), Y1, Y2
	VMULPD  32(AX), Y1, Y3
	VMULPD  64(AX), Y1, Y4
	VMULPD  96(AX), Y1, Y5
	VMULPD  128(AX), Y1, Y6
	VMULPD  160(AX), Y1, Y7
	VMULPD  192(AX), Y1, Y8
	VMULPD  224(AX), Y1, Y9
	VMULPD  256(AX), Y1, Y10
	VMULPD  288(AX), Y1, Y11
	VMULPD  320(AX), Y1, Y12
	VMULPD  352(AX), Y1, Y13
	VMULPD  384(AX), Y1, Y14
	VMULPD  416(AX), Y1, Y15
	VMOVUPD Y2, (CX)
	VMOVUPD Y3, 32(CX)
	VMOVUPD Y4, 64(CX)
	VMOVUPD Y5, 96(CX)
	VMOVUPD Y6, 128(CX)
	VMOVUPD Y7, 160(CX)
	VMOVUPD Y8, 192(CX)
	VMOVUPD Y9, 224(CX)
	VMOVUPD Y10, 256(CX)
	VMOVUPD Y11, 288(CX)
	VMOVUPD Y12, 320(CX)
	VMOVUPD Y13, 352(CX)
	VMOVUPD Y14, 384(CX)
	VMOVUPD Y15, 416(CX)
	ADDQ    $0x000001c0, AX
	ADDQ    $0x000001c0, CX
	SUBQ    $0x00000038, DX
	JMP     unrolledLoop14

unrolledLoop8:
	CMPQ    DX, $0x00000020
	JL      unrolledLoop4
	VMULPD  (AX), Y1, Y2
	VMULPD  32(AX), Y1, Y3
	VMULPD  64(AX), Y1, Y4
	VMULPD  96(AX), Y1, Y5
	VMULPD  128(AX), Y1, Y6
	VMULPD  160(AX), Y1, Y7
	VMULPD  192(AX), Y1, Y8
	VMULPD  224(AX), Y1, Y9
	VMOVUPD Y2, (CX)
	VMOVUPD Y3, 32(CX)
	VMOVUPD Y4, 64(CX)
	VMOVUPD Y5, 96(CX)
	VMOVUPD Y6, 128(CX)
	VMOVUPD Y7, 160(CX)
	VMOVUPD Y8, 192(CX)
	VMOVUPD Y9, 224(CX)
	ADDQ    $0x00000100, AX
	ADDQ    $0x00000100, CX
	SUBQ    $0x00000020, DX
	JMP     unrolledLoop8

unrolledLoop4:
	CMPQ    DX, $0x00000010
	JL      unrolledLoop1
	VMULPD  (AX), Y1, Y2
	VMULPD  32(AX), Y1, Y3
	VMULPD  64(AX), Y1, Y4
	VMULPD  96(AX), Y1, Y5
	VMOVUPD Y2, (CX)
	VMOVUPD Y3, 32(CX)
	VMOVUPD Y4, 64(CX)
	VMOVUPD Y5, 96(CX)
	ADDQ    $0x00000080, AX
	ADDQ    $0x00000080, CX
	SUBQ    $0x00000010, DX
	JMP     unrolledLoop4

unrolledLoop1:
	CMPQ    DX, $0x00000004
	JL      tailLoop
	VMULPD  (AX), Y1, Y2
	VMOVUPD Y2, (CX)
	ADDQ    $0x00000020, AX
	ADDQ    $0x00000020, CX
	SUBQ    $0x00000004, DX
	JMP     unrolledLoop1

tailLoop:
	CMPQ  DX, $0x00000000
	JE    end
	MOVSD (AX), X1
	MULSD X0, X1
	MOVSD X1, (CX)
	ADDQ  $0x00000008, AX
	ADDQ  $0x00000008, CX
	DECQ  DX
	JMP   tailLoop

end:
	RET

// func MulConstSSE32(c float32, x []float32, y []float32)
// Requires: SSE
TEXT ·MulConstSSE32(SB), NOSPLIT, $0-56
	MOVSS  c+0(FP), X0
	MOVQ   x_base+8(FP), AX
	MOVQ   y_base+32(FP), CX
	MOVQ   x_len+16(FP), DX
	SHUFPS $0x00, X0, X0

unrolledLoop14:
	CMPQ   DX, $0x00000038
	JL     unrolledLoop8
	MOVUPS (AX), X1
	MOVUPS 16(AX), X2
	MOVUPS 32(AX), X3
	MOVUPS 48(AX), X4
	MOVUPS 64(AX), X5
	MOVUPS 80(AX), X6
	MOVUPS 96(AX), X7
	MOVUPS 112(AX), X8
	MOVUPS 128(AX), X9
	MOVUPS 144(AX), X10
	MOVUPS 160(AX), X11
	MOVUPS 176(AX), X12
	MOVUPS 192(AX), X13
	MOVUPS 208(AX), X14
	MULPS  X0, X1
	MULPS  X0, X2
	MULPS  X0, X3
	MULPS  X0, X4
	MULPS  X0, X5
	MULPS  X0, X6
	MULPS  X0, X7
	MULPS  X0, X8
	MULPS  X0, X9
	MULPS  X0, X10
	MULPS  X0, X11
	MULPS  X0, X12
	MULPS  X0, X13
	MULPS  X0, X14
	MOVUPS X1, (CX)
	MOVUPS X2, 16(CX)
	MOVUPS X3, 32(CX)
	MOVUPS X4, 48(CX)
	MOVUPS X5, 64(CX)
	MOVUPS X6, 80(CX)
	MOVUPS X7, 96(CX)
	MOVUPS X8, 112(CX)
	MOVUPS X9, 128(CX)
	MOVUPS X10, 144(CX)
	MOVUPS X11, 160(CX)
	MOVUPS X12, 176(CX)
	MOVUPS X13, 192(CX)
	MOVUPS X14, 208(CX)
	ADDQ   $0x000000e0, AX
	ADDQ   $0x000000e0, CX
	SUBQ   $0x00000038, DX
	JMP    unrolledLoop14

unrolledLoop8:
	CMPQ   DX, $0x00000020
	JL     unrolledLoop4
	MOVUPS (AX), X1
	MOVUPS 16(AX), X2
	MOVUPS 32(AX), X3
	MOVUPS 48(AX), X4
	MOVUPS 64(AX), X5
	MOVUPS 80(AX), X6
	MOVUPS 96(AX), X7
	MOVUPS 112(AX), X8
	MULPS  X0, X1
	MULPS  X0, X2
	MULPS  X0, X3
	MULPS  X0, X4
	MULPS  X0, X5
	MULPS  X0, X6
	MULPS  X0, X7
	MULPS  X0, X8
	MOVUPS X1, (CX)
	MOVUPS X2, 16(CX)
	MOVUPS X3, 32(CX)
	MOVUPS X4, 48(CX)
	MOVUPS X5, 64(CX)
	MOVUPS X6, 80(CX)
	MOVUPS X7, 96(CX)
	MOVUPS X8, 112(CX)
	ADDQ   $0x00000080, AX
	ADDQ   $0x00000080, CX
	SUBQ   $0x00000020, DX
	JMP    unrolledLoop8

unrolledLoop4:
	CMPQ   DX, $0x00000010
	JL     unrolledLoop1
	MOVUPS (AX), X1
	MOVUPS 16(AX), X2
	MOVUPS 32(AX), X3
	MOVUPS 48(AX), X4
	MULPS  X0, X1
	MULPS  X0, X2
	MULPS  X0, X3
	MULPS  X0, X4
	MOVUPS X1, (CX)
	MOVUPS X2, 16(CX)
	MOVUPS X3, 32(CX)
	MOVUPS X4, 48(CX)
	ADDQ   $0x00000040, AX
	ADDQ   $0x00000040, CX
	SUBQ   $0x00000010, DX
	JMP    unrolledLoop4

unrolledLoop1:
	CMPQ   DX, $0x00000004
	JL     tailLoop
	MOVUPS (AX), X1
	MULPS  X0, X1
	MOVUPS X1, (CX)
	ADDQ   $0x00000010, AX
	ADDQ   $0x00000010, CX
	SUBQ   $0x00000004, DX
	JMP    unrolledLoop1

tailLoop:
	CMPQ  DX, $0x00000000
	JE    end
	MOVSS (AX), X1
	MULSS X0, X1
	MOVSS X1, (CX)
	ADDQ  $0x00000004, AX
	ADDQ  $0x00000004, CX
	DECQ  DX
	JMP   tailLoop

end:
	RET

// func MulConstSSE64(c float64, x []float64, y []float64)
// Requires: SSE2
TEXT ·MulConstSSE64(SB), NOSPLIT, $0-56
	MOVSD  c+0(FP), X0
	MOVQ   x_base+8(FP), AX
	MOVQ   y_base+32(FP), CX
	MOVQ   x_len+16(FP), DX
	SHUFPD $0x00, X0, X0

unrolledLoop14:
	CMPQ   DX, $0x0000001c
	JL     unrolledLoop8
	MOVUPD (AX), X1
	MOVUPD 16(AX), X2
	MOVUPD 32(AX), X3
	MOVUPD 48(AX), X4
	MOVUPD 64(AX), X5
	MOVUPD 80(AX), X6
	MOVUPD 96(AX), X7
	MOVUPD 112(AX), X8
	MOVUPD 128(AX), X9
	MOVUPD 144(AX), X10
	MOVUPD 160(AX), X11
	MOVUPD 176(AX), X12
	MOVUPD 192(AX), X13
	MOVUPD 208(AX), X14
	MULPD  X0, X1
	MULPD  X0, X2
	MULPD  X0, X3
	MULPD  X0, X4
	MULPD  X0, X5
	MULPD  X0, X6
	MULPD  X0, X7
	MULPD  X0, X8
	MULPD  X0, X9
	MULPD  X0, X10
	MULPD  X0, X11
	MULPD  X0, X12
	MULPD  X0, X13
	MULPD  X0, X14
	MOVUPD X1, (CX)
	MOVUPD X2, 16(CX)
	MOVUPD X3, 32(CX)
	MOVUPD X4, 48(CX)
	MOVUPD X5, 64(CX)
	MOVUPD X6, 80(CX)
	MOVUPD X7, 96(CX)
	MOVUPD X8, 112(CX)
	MOVUPD X9, 128(CX)
	MOVUPD X10, 144(CX)
	MOVUPD X11, 160(CX)
	MOVUPD X12, 176(CX)
	MOVUPD X13, 192(CX)
	MOVUPD X14, 208(CX)
	ADDQ   $0x000000e0, AX
	ADDQ   $0x000000e0, CX
	SUBQ   $0x0000001c, DX
	JMP    unrolledLoop14

unrolledLoop8:
	CMPQ   DX, $0x00000010
	JL     unrolledLoop4
	MOVUPD (AX), X1
	MOVUPD 16(AX), X2
	MOVUPD 32(AX), X3
	MOVUPD 48(AX), X4
	MOVUPD 64(AX), X5
	MOVUPD 80(AX), X6
	MOVUPD 96(AX), X7
	MOVUPD 112(AX), X8
	MULPD  X0, X1
	MULPD  X0, X2
	MULPD  X0, X3
	MULPD  X0, X4
	MULPD  X0, X5
	MULPD  X0, X6
	MULPD  X0, X7
	MULPD  X0, X8
	MOVUPD X1, (CX)
	MOVUPD X2, 16(CX)
	MOVUPD X3, 32(CX)
	MOVUPD X4, 48(CX)
	MOVUPD X5, 64(CX)
	MOVUPD X6, 80(CX)
	MOVUPD X7, 96(CX)
	MOVUPD X8, 112(CX)
	ADDQ   $0x00000080, AX
	ADDQ   $0x00000080, CX
	SUBQ   $0x00000010, DX
	JMP    unrolledLoop8

unrolledLoop4:
	CMPQ   DX, $0x00000008
	JL     unrolledLoop1
	MOVUPD (AX), X1
	MOVUPD 16(AX), X2
	MOVUPD 32(AX), X3
	MOVUPD 48(AX), X4
	MULPD  X0, X1
	MULPD  X0, X2
	MULPD  X0, X3
	MULPD  X0, X4
	MOVUPD X1, (CX)
	MOVUPD X2, 16(CX)
	MOVUPD X3, 32(CX)
	MOVUPD X4, 48(CX)
	ADDQ   $0x00000040, AX
	ADDQ   $0x00000040, CX
	SUBQ   $0x00000008, DX
	JMP    unrolledLoop4

unrolledLoop1:
	CMPQ   DX, $0x00000002
	JL     tailLoop
	MOVUPD (AX), X1
	MULPD  X0, X1
	MOVUPD X1, (CX)
	ADDQ   $0x00000010, AX
	ADDQ   $0x00000010, CX
	SUBQ   $0x00000002, DX
	JMP    unrolledLoop1

tailLoop:
	CMPQ  DX, $0x00000000
	JE    end
	MOVSD (AX), X1
	MULSD X0, X1
	MOVSD X1, (CX)
	ADDQ  $0x00000008, AX
	ADDQ  $0x00000008, CX
	DECQ  DX
	JMP   tailLoop

end:
	RET