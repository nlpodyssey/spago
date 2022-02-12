#include "textflag.h"

// func Log(x float64) float64
TEXT ·Log(SB),NOSPLIT,$0
	B ·log(SB)

TEXT ·Remainder(SB),NOSPLIT,$0
	B ·remainder(SB)
