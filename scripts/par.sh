# A script's compile steps at once, sourced: `par CMD ...` starts one, `par_wait` waits for all of them and fails if any failed.
# A script has a handful, below the 14 jobs the host's builds may take (Makefile's JOBS).
par_pids=()
par() {
    "$@" &
    par_pids+=($!)
}
par_wait() {
    local rc=0 p
    for p in "${par_pids[@]}"; do
        wait "$p" || rc=1
    done
    par_pids=()
    return "$rc"
}
