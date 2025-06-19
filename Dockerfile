# Build stage
FROM golang:1.23.0-alpine AS builder

WORKDIR /app

COPY . .

RUN go build github.com/fxnlabs/function-node/cmd/node

FROM alpine:3.22.0

WORKDIR /app

COPY --from=builder /app/node .
COPY --from=builder /app/fixtures/abi ./fixtures/abi

CMD ["./node"]