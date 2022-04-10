package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestVariable_NewVariable(t *testing.T) {
	t.Run("float32", testVariableNewVariable[float32])
	t.Run("float64", testVariableNewVariable[float64])
}

func testVariableNewVariable[T mat.DType](t *testing.T) {
	t.Run("with requiresGrad true", func(t *testing.T) {
		s := mat.NewScalar[T](1)
		v := NewVariable[T](s, true)
		assert.NotNil(t, v)
		assert.Same(t, s, v.Value())
		assert.True(t, v.RequiresGrad())
	})

	t.Run("with requiresGrad false", func(t *testing.T) {
		s := mat.NewScalar[T](1)
		v := NewVariable[T](s, false)
		assert.NotNil(t, v)
		assert.Same(t, s, v.Value())
		assert.False(t, v.RequiresGrad())
	})
}

func TestVariable_NewScalar(t *testing.T) {
	t.Run("float32", testVariableNewScalar[float32])
	t.Run("float64", testVariableNewScalar[float64])
}

func testVariableNewScalar[T mat.DType](t *testing.T) {
	s := NewScalar[T](42)
	assert.NotNil(t, s)
	assert.False(t, s.RequiresGrad())
	v := s.Value()
	assert.NotNil(t, v)
	assert.True(t, mat.IsScalar(v))
	assert.Equal(t, T(42.0), v.Scalar())
}

func TestVariable_Constant(t *testing.T) {
	t.Run("float32", testVariableConstant[float32])
	t.Run("float64", testVariableConstant[float64])
}

func testVariableConstant[T mat.DType](t *testing.T) {
	c := Constant[T](42)
	assert.NotNil(t, c)
	assert.False(t, c.RequiresGrad())
	v := c.Value()
	assert.NotNil(t, v)
	assert.True(t, mat.IsScalar(v))
	assert.Equal(t, T(42.0), v.Scalar())
}
