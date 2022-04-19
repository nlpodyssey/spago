// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

import (
	"fmt"
	"unsafe"

	"github.com/awalterschulze/gographviz"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

type builder[T mat.DType] struct {
	// List of unique nodes
	nodes []ag.Node[T]

	gv gographviz.Interface

	// showNodesWithoutEdges indicates whether to show graph nodes
	// which have no connections.
	showNodesWithoutEdges bool
}

// Option allows to tweak the graphviz generation.
type Option[T mat.DType] func(*builder[T])

// ShowNodesWithoutEdges enables whether to show graph nodes
// which have no connections.
func ShowNodesWithoutEdges[T mat.DType](value bool) Option[T] {
	return func(m *builder[T]) {
		m.showNodesWithoutEdges = value
	}
}

func newBuilder[T mat.DType](outputNodes []ag.Node[T], options ...Option[T]) *builder[T] {
	nodes := uniqueNodes(outputNodes)
	b := &builder[T]{
		nodes: nodes,
		gv:    gographviz.NewEscape(),
	}
	for _, option := range options {
		option(b)
	}
	return b
}

func uniqueNodes[T mat.DType](outputNodes []ag.Node[T]) []ag.Node[T] {
	visited := make(map[ag.Node[T]]struct{})
	for _, node := range outputNodes {
		visitUniqueNodes(visited, node)
	}

	nodes := make([]ag.Node[T], 0, len(visited))
	for n := range visited {
		nodes = append(nodes, n)
	}
	return nodes
}

func visitUniqueNodes[T mat.DType](visited map[ag.Node[T]]struct{}, node ag.Node[T]) {
	if _, ok := visited[node]; ok {
		return
	}
	visited[node] = struct{}{}

	if op, ok := node.(*ag.Operator[T]); ok {
		for _, operand := range op.Operands() {
			visitUniqueNodes[T](visited, operand)
		}
	}
}

func (b *builder[T]) build() (gographviz.Interface, error) {
	if err := b.gv.SetDir(true); err != nil {
		return nil, err
	}
	if err := b.gv.AddAttr("", "rankdir", "LR"); err != nil {
		return nil, err
	}

	var nodesWithoutEdges map[ag.Node[T]]struct{}
	if !b.showNodesWithoutEdges {
		nodesWithoutEdges = b.findNodesWithoutEdges()
	}

	for _, node := range b.nodes {
		if _, ok := nodesWithoutEdges[node]; ok {
			continue
		}
		switch nt := node.(type) {
		case *ag.Variable[T]:
			if err := b.addVariable(nt); err != nil {
				return nil, err
			}
		case *ag.Operator[T]:
			if err := b.addOperator(nt); err != nil {
				return nil, err
			}
		case *ag.Wrapper[T]:
			if err := b.addWrapper(nt); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unexpected node type %T", node)
		}
	}
	return b.gv, nil
}

func (b *builder[T]) addVariable(v *ag.Variable[T]) error {
	id := fmt.Sprintf("%d", uintptr(unsafe.Pointer(v)))
	name := v.Name()
	if name == "" {
		name = "-"
	}
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%s</FONT><BR />
			variable <B>%s</B><BR />
			%s
		>`,
		id,
		name,
		matrixShapeString(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return b.gv.AddNode("", id, attrs)
}

func (b *builder[T]) addWrapper(v *ag.Wrapper[T]) error {
	if param, ok := v.Node.(nn.Param[T]); ok {
		return b.addParam(v, param.Name())
	}

	id := fmt.Sprintf("%d", uintptr(unsafe.Pointer(v)))
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%s</FONT><BR />
			wrapper<BR />
			%s
		>`,
		id,
		matrixShapeString(v.Node.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return b.gv.AddNode("", id, attrs)
}

func (b *builder[T]) addParam(v *ag.Wrapper[T], name string) error {
	name1 := name
	if name1 == "" {
		name1 = "-"
	}
	id := fmt.Sprintf("%d", uintptr(unsafe.Pointer(v)))
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%s</FONT><BR />
			param <B>%s</B><BR />
			%s
		>`,
		id,
		name1,
		matrixShapeString(v.Node.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
	}
	return b.gv.AddNode("", id, attrs)
}

func (b *builder[T]) addOperator(op *ag.Operator[T]) error {
	operatorID := fmt.Sprintf("%d", uintptr(unsafe.Pointer(op)))

	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%s</FONT><BR />
			<B>%s</B><BR />
			%s
		>`,
		operatorID,
		op.Name(),
		matrixShapeString(op.Value()),
	)
	attrs := map[string]string{
		"label": label,
	}
	if err := b.gv.AddNode("", operatorID, attrs); err != nil {
		return err
	}

	for _, operand := range op.Operands() {
		operandID := fmt.Sprintf("%d", uintptr(unsafe.Pointer(&operand)))
		if err := b.gv.AddEdge(operandID, operatorID, true, nil); err != nil {
			return err
		}
	}
	return nil
}

func (b *builder[T]) findNodesWithoutEdges() map[ag.Node[T]]struct{} {
	ids := make(map[ag.Node[T]]struct{})
	for _, node := range b.nodes {
		operator, isOperator := node.(*ag.Operator[T])
		if !isOperator || len(operator.Operands()) == 0 {
			ids[node] = struct{}{}
			continue
		}
		for _, operand := range operator.Operands() {
			delete(ids, operand)
		}
	}
	return ids
}

func matrixShapeString[T mat.DType](m mat.Matrix[T]) string {
	if m == nil {
		return ""
	}
	if mat.IsScalar(m) {
		return "scalar"
	}
	return fmt.Sprintf("%d × %d", m.Rows(), m.Columns())
}
