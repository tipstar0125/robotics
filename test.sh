export RUSTFLAGS=-Awarnings
cargo test -r --features local --bin $1 -- --nocapture