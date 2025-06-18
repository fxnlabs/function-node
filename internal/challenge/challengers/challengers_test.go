package challengers

import (
	"net/http"

	"github.com/stretchr/testify/assert"
)

type mockRoundTripper struct {
	roundTripFunc func(req *http.Request) (*http.Response, error)
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.roundTripFunc(req)
}

func newTestClient(fn func(req *http.Request) (*http.Response, error)) *http.Client {
	return &http.Client{
		Transport: &mockRoundTripper{
			roundTripFunc: fn,
		},
	}
}

type errorReader struct{}

func (r *errorReader) Read(p []byte) (n int, err error) {
	return 0, assert.AnError
}
