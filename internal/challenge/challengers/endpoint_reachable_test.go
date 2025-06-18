package challengers

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func TestNewEndpointReachableChallenger(t *testing.T) {
	challenger := NewEndpointReachableChallenger()
	assert.NotNil(t, challenger)
	assert.NotNil(t, challenger.Client)
}

func TestEndpointReachableChallenger_Execute(t *testing.T) {
	log := zap.NewNop()

	t.Run("reachable endpoint", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       http.NoBody,
			}, nil
		})
		challenger := &EndpointReachableChallenger{Client: mockClient}
		result, err := challenger.Execute("http://example.com", log)
		assert.NoError(t, err)
		assert.Equal(t, map[string]bool{"reachable": true}, result)
	})

	t.Run("unreachable endpoint", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusNotFound,
				Body:       http.NoBody,
			}, nil
		})
		challenger := &EndpointReachableChallenger{Client: mockClient}
		result, err := challenger.Execute("http://example.com", log)
		assert.NoError(t, err)
		assert.Equal(t, map[string]bool{"reachable": false}, result)
	})

	t.Run("http get error", func(t *testing.T) {
		mockClient := newTestClient(func(req *http.Request) (*http.Response, error) {
			return nil, assert.AnError
		})
		challenger := &EndpointReachableChallenger{Client: mockClient}
		result, err := challenger.Execute("http://example.com", log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})

	t.Run("invalid payload", func(t *testing.T) {
		challenger := &EndpointReachableChallenger{Client: &http.Client{}}
		result, err := challenger.Execute(123, log)
		assert.Error(t, err)
		assert.Nil(t, result)
	})
}
