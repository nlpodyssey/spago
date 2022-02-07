// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define	Ln2Hi	6.9313812256e-01
#define	Ln2Lo	9.0580006145e-06
#define	Log2e	1.4426950216e+00
#define	Overflow	7.097827e+02
#define	Underflow	-7.451332e+02
#define	Overflow2	1.024000e+03
#define	Underflow2	-1.0740e+03
#define	NearZero	0x317fffff	// 2**-28
#define	PosInf	0x7f800000
#define	FracMask	0x07fffff
#define	C1	0x34000000	// 2**-23
#define	P1	1.6666667163e-01	// 0x3FC55555; 0x55555555
#define	P2	-2.7777778450e-03	// 0xBF66C16C; 0x16BEBD93
#define	P3	6.6137559770e-05	// 0x3F11566A; 0xAF25DE2C
#define	P4	-1.6533901999e-06	// 0xBEBBBD41; 0xC5D26BF1
#define	P5	4.1381369442e-08	// 0x3E663769; 0x72BEA4D0

// Exp returns e**x, the base-e exponential of x.
// This is an assembly implementation of the method used for function Exp in file exp.go.
//
// func Exp(x float32) float32
TEXT ·Exp(SB),$0-16
	FMOVS	x+0(FP), F0	// F0 = x
	FCMPS	F0, F0
	BNE	isNaN		// x = NaN, return NaN
	FMOVS	$Overflow, F1
	FCMPS	F1, F0
	BGT	overflow	// x > Overflow, return PosInf
	FMOVS	$Underflow, F1
	FCMPS	F1, F0
	BLT	underflow	// x < Underflow, return 0
	MOVW	$NearZero, R0
	FMOVS	R0, F2
	FABSS	F0, F3
	FMOVS	$1.0, F1	// F1 = 1.0
	FCMPS	F2, F3
	BLT	nearzero	// fabs(x) < NearZero, return 1 + x
	// argument reduction, x = k*ln2 + r,  |r| <= 0.5*ln2
	// computed as r = hi - lo for extra precision.
	FMOVS	$Log2e, F2
	FMOVS	$0.5, F3
	FNMSUBS	F0, F3, F2, F4	// Log2e*x - 0.5
	FMADDS	F0, F3, F2, F3	// Log2e*x + 0.5
	FCMPS	$0.0, F0
	FCSELS	LT, F4, F3, F3	// F3 = k
	FCVTZSS	F3, R1		// R1 = int(k)
	SCVTFS	R1, F3		// F3 = float32(int(k))
	FMOVS	$Ln2Hi, F4	// F4 = Ln2Hi
	FMOVS	$Ln2Lo, F5	// F5 = Ln2Lo
	FMSUBS	F3, F0, F4, F4	// F4 = hi = x - float32(int(k))*Ln2Hi
	FMULS	F3, F5		// F5 = lo = float32(int(k)) * Ln2Lo
	FSUBS	F5, F4, F6	// F6 = r = hi - lo
	FMULS	F6, F6, F7	// F7 = t = r * r
	// compute y
	FMOVS	$P5, F8		// F8 = P5
	FMOVS	$P4, F9		// F9 = P4
	FMADDS	F7, F9, F8, F13	// P4+t*P5
	FMOVS	$P3, F10	// F10 = P3
	FMADDS	F7, F10, F13, F13	// P3+t*(P4+t*P5)
	FMOVS	$P2, F11	// F11 = P2
	FMADDS	F7, F11, F13, F13	// P2+t*(P3+t*(P4+t*P5))
	FMOVS	$P1, F12	// F12 = P1
	FMADDS	F7, F12, F13, F13	// P1+t*(P2+t*(P3+t*(P4+t*P5)))
	FMSUBS	F7, F6, F13, F13	// F13 = c = r - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
	FMOVS	$2.0, F14
	FSUBS	F13, F14
	FMULS	F6, F13, F15
	FDIVS	F14, F15	// F15 = (r*c)/(2-c)
	FSUBS	F15, F5, F15	// lo-(r*c)/(2-c)
	FSUBS	F4, F15, F15	// (lo-(r*c)/(2-c))-hi
	FSUBS	F15, F1, F16	// F16 = y = 1-((lo-(r*c)/(2-c))-hi)
	// inline Ldexp(y, k), benefit:
	// 1, no parameter pass overhead.
	// 2, skip unnecessary checks for Inf/NaN/Zero
	FMOVS	F16, R0
	ANDS	$FracMask, R0, R2	// fraction
	LSRW	$23, R0, R5	// exponent
	ADDS	R1, R5		// R1 = int(k)
	CMPW	$1, R5
	BGE	normal
	ADDS	$23, R5		// denormal
	MOVW	$C1, R8
	FMOVS	R8, F1		// m = 2**-23
normal:
	ORRW	R5<<23, R2, R0
	FMOVS	R0, F0
	FMULS	F1, F0		// return m * x
	FMOVS	F0, ret+8(FP)
	RET
nearzero:
	FADDS	F1, F0
isNaN:
	FMOVS	F0, ret+8(FP)
	RET
underflow:
	MOVW	ZR, ret+8(FP)
	RET
overflow:
	MOVW	$PosInf, R0
	MOVW	R0, ret+8(FP)
	RET
