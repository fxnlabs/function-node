# Build stage
FROM golang:1.23.0-alpine AS builder

WORKDIR /app

COPY . .

RUN go build github.com/fxnlabs/function-node/cmd/fxn

FROM alpine:3.22.0

WORKDIR /app

COPY --from=builder /app/fxn .

CMD ["./fxn"]