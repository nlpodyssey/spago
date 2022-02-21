package binder

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

// BindNode returns a Node that satisfies the same interface
// of the one in input but connected to the given graph
// TODO: fix this without direct reference to `nn.Param`
func BindNode[T mat.DType, P ag.Node[T]](g *ag.Graph[T], p ag.Node[T]) ag.Node[T] {
	if _, ok := p.(*nn.ParamNode[T]); ok {
		panic("nn: impossible to bind a param node.")
	}
	switch p := p.(type) {
	case nn.Param[T]:
		if p.RequiresGrad() {
			return &nn.ParamNode[T]{Param: p, Node: g.NewWrap(p)}
		}
		return &nn.ParamNode[T]{Param: p, Node: g.NewWrapNoGrad(p)}
	default:
		panic("invalid node type")
	}
}
