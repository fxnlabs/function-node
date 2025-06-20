package metrics

import (
	"net/http"
	"strconv"
)

// ResponseWriterInterceptor is a wrapper around http.ResponseWriter to capture the status code.
type ResponseWriterInterceptor struct {
	http.ResponseWriter
	StatusCode int
}

// NewResponseWriterInterceptor creates a new ResponseWriterInterceptor.
func NewResponseWriterInterceptor(w http.ResponseWriter) *ResponseWriterInterceptor {
	// Default to 200 OK if WriteHeader is not called.
	return &ResponseWriterInterceptor{w, http.StatusOK}
}

// WriteHeader captures the status code and calls the original WriteHeader.
func (rwi *ResponseWriterInterceptor) WriteHeader(code int) {
	rwi.StatusCode = code
	rwi.ResponseWriter.WriteHeader(code)
}

// Middleware wraps an http.Handler to record endpoint responses.
func Middleware(next http.Handler, endpointPath string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		interceptor := NewResponseWriterInterceptor(w)
		next.ServeHTTP(interceptor, r)
		EndpointResponses.WithLabelValues(endpointPath, strconv.Itoa(interceptor.StatusCode)).Inc()
	})
}
