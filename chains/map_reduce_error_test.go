package chains

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
)

type mapReduceResultChain struct {
	result map[string]any
	err    error
}

func (c mapReduceResultChain) Call(context.Context, map[string]any, ...ChainCallOption) (map[string]any, error) {
	return c.result, c.err
}

func (c mapReduceResultChain) GetMemory() schema.Memory { return memory.NewSimple() }
func (c mapReduceResultChain) GetInputKeys() []string   { return []string{"input_documents"} }
func (c mapReduceResultChain) GetOutputKeys() []string  { return []string{"text"} }

func TestMapReduceReduceError(t *testing.T) {
	t.Parallel()
	reduceErr := errors.New("reduce failed")
	for _, tc := range []struct {
		name              string
		result            map[string]any
		err               error
		intermediateSteps bool
	}{
		{name: "nil result with intermediate steps", err: reduceErr, intermediateSteps: true},
		{name: "nil result without intermediate steps", err: reduceErr},
		{name: "partial result with intermediate steps", result: map[string]any{"text": "partial"}, err: reduceErr, intermediateSteps: true},
		{name: "partial result without intermediate steps", result: map[string]any{"text": "partial"}, err: reduceErr},
		{name: "success with intermediate steps", result: map[string]any{"text": "complete"}, intermediateSteps: true},
		{name: "success without intermediate steps", result: map[string]any{"text": "complete"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			chain := NewMapReduceDocuments(
				NewLLMChain(
					&testLanguageModel{expResult: "mapped"},
					prompts.NewPromptTemplate("{{.context}}", []string{"context"}),
				),
				mapReduceResultChain{result: tc.result, err: tc.err},
			)
			chain.ReturnIntermediateSteps = tc.intermediateSteps
			var result map[string]any
			var err error
			require.NotPanics(t, func() {
				result, err = Call(context.Background(), chain, map[string]any{
					"input_documents": []schema.Document{{PageContent: "source"}},
				})
			})
			if tc.err != nil {
				require.ErrorIs(t, err, tc.err)
			} else {
				require.NoError(t, err)
			}
			if tc.result == nil {
				require.Nil(t, result)
				return
			}
			require.Equal(t, tc.result["text"], result["text"])
			if tc.intermediateSteps {
				require.Equal(t, []map[string]any{{"text": "mapped"}}, result[_intermediateStepsOutputKey])
			} else {
				require.NotContains(t, result, _intermediateStepsOutputKey)
			}
		})
	}
}
