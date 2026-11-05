# Intel-Phi-QwenGPU: a SIMT GPU compiled into Qwen3.8-35B's weights, Qwen3.8-2B running on it, on the Xeon Phi 7120A.
# Host binaries go to out/host; everything is rebuilt from src/. Every build runs 14 jobs (the 5800X has 16 threads, 2 stay
# with the desktop): `make JOBS=n` overrides it, and so does a -j on make's command line; the tool shell's `make` is an
# alias for `make -j16`, so call /usr/bin/make there.
CC       := clang
CXX      := clang++
CFLAGS   := -std=c11 -O2 -g -Wall -Wextra -Werror -D_GNU_SOURCE -ffp-contract=off
CXXFLAGS := -std=c++20 -O2 -g -Wall -Wextra -Werror -D_GNU_SOURCE -ffp-contract=off -mfma -Wno-deprecated-anon-enum-enum-conversion
OUT      := out/host
JOBS     := 14
MAKEFLAGS += -j$(JOBS) -Oline
ifneq ($(filter clean,$(MAKECMDGOALS)),)
.NOTPARALLEL:
endif
$(shell mkdir -p $(OUT))

GPU_HDR := src/gpu/sm.h src/gpu/sm_ref.h

.PHONY: help all compdb clean
help:
	@echo "make all      build the host tools ($(OUT))"
	@echo "make compdb   write compile_commands.json and run clangd over the tree"
	@echo "make clean    remove out/host"

all: $(OUT)/smtest $(OUT)/ptxdump $(OUT)/ptxrun

LOWER_HDR := src/gpu/kinstr.h src/gpu/blocks.h src/gpu/lower.h src/gpu/ptx.h src/gpu/sm.h

$(OUT)/sm_ref.o: src/gpu/sm_ref.c $(GPU_HDR)
	$(CC) $(CFLAGS) -c -o $@ src/gpu/sm_ref.c

$(OUT)/ptx.o: src/gpu/ptx.cc src/gpu/ptx.h
	$(CXX) $(CXXFLAGS) -c -o $@ src/gpu/ptx.cc

$(OUT)/blocks.o: src/gpu/blocks.cc $(LOWER_HDR)
	$(CXX) $(CXXFLAGS) -c -o $@ src/gpu/blocks.cc

$(OUT)/lower.o: src/gpu/lower.cc $(LOWER_HDR)
	$(CXX) $(CXXFLAGS) -c -o $@ src/gpu/lower.cc

$(OUT)/ptxrun: src/tools/ptxrun.cc $(OUT)/ptx.o $(OUT)/blocks.o $(OUT)/lower.o $(OUT)/sm_ref.o $(LOWER_HDR) src/gpu/sm_ref.h
	$(CXX) $(CXXFLAGS) -o $@ src/tools/ptxrun.cc $(OUT)/ptx.o $(OUT)/blocks.o $(OUT)/lower.o $(OUT)/sm_ref.o

$(OUT)/smtest: tests/gpu/smtest.c $(OUT)/sm_ref.o $(GPU_HDR)
	$(CC) $(CFLAGS) -o $@ tests/gpu/smtest.c $(OUT)/sm_ref.o

$(OUT)/ptxdump: src/tools/ptxdump.cc $(OUT)/ptx.o src/gpu/ptx.h
	$(CXX) $(CXXFLAGS) -o $@ src/tools/ptxdump.cc $(OUT)/ptx.o

compdb:
	scripts/compdb.sh

clean:
	rm -rf $(OUT)

# --- the text path: the 2B's exact and integer references, the tokenizer (host, test references only) ---------------------
TEXT_HDR := src/text/gguf.h src/text/tok.h
TEXT_OBJ := $(OUT)/gguf.o $(OUT)/tok.o
# the reference fuses by instruction, never through libm; -ffp-contract=off keeps it to the written FMAs
REFFLAGS := $(CXXFLAGS) -O3 -mavx2 -Wno-unknown-pragmas -Wno-deprecated-enum-float-conversion
$(OUT)/gguf.o: src/text/gguf.cc src/text/gguf.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<
$(OUT)/tok.o: src/text/tok.cc $(TEXT_HDR) src/text/unicode_tab.inc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
$(OUT)/q35_ref.o: src/text/q35_ref.cc src/text/q35.h src/text/q35_hash.h src/text/elem.h src/text/gguf.h
	$(CXX) $(REFFLAGS) -c -o $@ $<
$(OUT)/q2b_ref.o: src/text/q2b_ref.cc src/text/q2b.h src/text/q35.h src/text/elem.h src/text/gguf.h
	$(CXX) $(REFFLAGS) -c -o $@ src/text/q2b_ref.cc
$(OUT)/q2b_fix.o: src/text/q2b_fix.cc src/text/q2b_fix.h src/text/q2b.h src/text/q35.h src/text/gguf.h
	$(CXX) $(REFFLAGS) -c -o $@ src/text/q2b_fix.cc
$(OUT)/q2b_fix_check: tests/text/q2b_fix_check.cc $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ) $(TEXT_HDR) src/text/q2b_fix.h src/text/q2b.h
	$(CXX) $(REFFLAGS) -o $@ tests/text/q2b_fix_check.cc $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ)
$(OUT)/gguf_meta: src/tools/gguf_meta.cc $(OUT)/gguf.o src/text/gguf.h
	$(CXX) $(CXXFLAGS) -o $@ src/tools/gguf_meta.cc $(OUT)/gguf.o

text: $(OUT)/q2b_fix_check $(OUT)/gguf_meta
.PHONY: text

# --- the GPU image of the 2B and its checker ------------------------------------------------------------------------------
$(OUT)/image.o: src/gpu/image.cc src/gpu/image.h src/gpu/sm_ref.h src/gpu/sm.h src/guest/q2b_map.h
	$(CXX) $(CXXFLAGS) -c -o $@ src/gpu/image.cc
$(OUT)/q2b_build: src/tools/q2b_build.cc $(OUT)/ptx.o $(OUT)/blocks.o $(OUT)/lower.o $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ) $(LOWER_HDR) src/guest/q2b_map.h src/text/q2b_fix.h
	$(CXX) $(REFFLAGS) -o $@ src/tools/q2b_build.cc $(OUT)/ptx.o $(OUT)/blocks.o $(OUT)/lower.o $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ)
gpu: $(OUT)/q2b_build
.PHONY: gpu
$(OUT)/q2b_gpu_check: src/tools/q2b_gpu_check.cc $(OUT)/image.o $(OUT)/sm_ref.o $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ) src/gpu/image.h src/guest/q2b_map.h src/text/q2b_fix.h
	$(CXX) $(REFFLAGS) -o $@ src/tools/q2b_gpu_check.cc $(OUT)/image.o $(OUT)/sm_ref.o $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ)
gpu: $(OUT)/q2b_gpu_check
$(OUT)/q2b_gpu_probe: src/tools/q2b_gpu_probe.cc $(OUT)/image.o $(OUT)/sm_ref.o $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ) src/gpu/image.h src/guest/q2b_map.h src/text/q2b_fix.h
	$(CXX) $(REFFLAGS) -o $@ src/tools/q2b_gpu_probe.cc $(OUT)/image.o $(OUT)/sm_ref.o $(OUT)/q2b_fix.o $(OUT)/q2b_ref.o $(OUT)/q35_ref.o $(TEXT_OBJ)
gpu: $(OUT)/q2b_gpu_probe

# --- libqgpucl: OpenCL 1.2 subset over the embedded GPU on the reference machine ----------------------------------------
$(OUT)/qgpucl.o: src/cl/qgpucl.cc src/gpu/image.h src/gpu/lower.h src/gpu/ptx.h src/gpu/sm_ref.h src/gpu/sm.h
	$(CXX) $(CXXFLAGS) -DQGPU_ROOT='"$(CURDIR)"' -c -o $@ src/cl/qgpucl.cc
$(OUT)/libqgpucl.a: $(OUT)/qgpucl.o $(OUT)/ptx.o $(OUT)/blocks.o $(OUT)/lower.o $(OUT)/sm_ref.o $(OUT)/image.o
	ar rcs $@ $^
$(OUT)/clhello: tests/cl/clhello.c $(OUT)/libqgpucl.a
	$(CC) $(CFLAGS) -o $@ tests/cl/clhello.c $(OUT)/libqgpucl.a -lstdc++ -lm
cl: $(OUT)/libqgpucl.a $(OUT)/clhello
.PHONY: cl
$(OUT)/q2b_cl: tests/cl/q2b_cl.c $(OUT)/libqgpucl.a src/cl/qgpucl_ext.h src/guest/q2b_map.h
	$(CC) $(CFLAGS) -Wno-deprecated-anon-enum-enum-conversion -o $@ tests/cl/q2b_cl.c $(OUT)/libqgpucl.a -lstdc++ -lm
cl: $(OUT)/q2b_cl

# --- the circuit: the machine as gates (G4), its exact evaluator, the gate against the reference's stream -------------
CIRC_HDR := src/circuit/expr.h src/circuit/graph.h src/circuit/gate.h
$(OUT)/graph.o: src/circuit/graph.cc $(CIRC_HDR)
	$(CXX) $(CXXFLAGS) -c -o $@ src/circuit/graph.cc
$(OUT)/gate.o: src/circuit/gate.cc $(CIRC_HDR) src/gpu/sm.h
	$(CXX) $(CXXFLAGS) -c -o $@ src/circuit/gate.cc
$(OUT)/eval.o: src/circuit/eval.cc src/circuit/eval.h $(CIRC_HDR) src/gpu/sm_ref.h src/gpu/sm.h
	$(CXX) $(CXXFLAGS) -c -o $@ src/circuit/eval.cc
$(OUT)/sm_circ.o: src/circuit/sm_circ.cc src/circuit/sm_circ.h $(CIRC_HDR) src/gpu/sm.h
	$(CXX) $(CXXFLAGS) -c -o $@ src/circuit/sm_circ.cc
$(OUT)/circ_gate: src/tools/circ_gate.cc $(OUT)/graph.o $(OUT)/gate.o $(OUT)/eval.o $(OUT)/sm_circ.o $(OUT)/image.o $(OUT)/sm_ref.o $(TEXT_OBJ) src/circuit/sm_circ.h src/circuit/eval.h src/gpu/image.h src/guest/q2b_map.h
	$(CXX) $(CXXFLAGS) -o $@ src/tools/circ_gate.cc $(OUT)/graph.o $(OUT)/gate.o $(OUT)/eval.o $(OUT)/sm_circ.o $(OUT)/image.o $(OUT)/sm_ref.o $(TEXT_OBJ)
circuit: $(OUT)/circ_gate
.PHONY: circuit

# --- the model file (G5): the emitter, the forward over the file alone, its gate against the reference, the runner -------
CIRC_OBJ := $(OUT)/graph.o $(OUT)/gate.o $(OUT)/eval.o $(OUT)/sm_circ.o
$(OUT)/tforward.o: src/exec/tforward.cc src/exec/tforward.h src/exec/membits.h src/text/gguf.h src/text/elem.h
	$(CXX) $(REFFLAGS) -c -o $@ src/exec/tforward.cc
$(OUT)/gpu_emit: src/circuit/emit.cc $(CIRC_OBJ) $(OUT)/image.o $(OUT)/sm_ref.o $(TEXT_OBJ) src/circuit/sm_circ.h src/circuit/eval.h src/exec/membits.h src/gpu/image.h
	$(CXX) $(CXXFLAGS) -o $@ src/circuit/emit.cc $(CIRC_OBJ) $(OUT)/image.o $(OUT)/sm_ref.o $(TEXT_OBJ)
$(OUT)/gpu_run: src/tools/gpu_run.cc $(OUT)/tforward.o $(TEXT_OBJ) src/exec/tforward.h src/guest/q2b_map.h src/text/tok.h
	$(CXX) $(REFFLAGS) -o $@ src/tools/gpu_run.cc $(OUT)/tforward.o $(TEXT_OBJ)
$(OUT)/emit_gate: src/tools/emit_gate.cc $(OUT)/tforward.o $(CIRC_OBJ) $(OUT)/image.o $(OUT)/sm_ref.o $(TEXT_OBJ) src/exec/tforward.h src/circuit/sm_circ.h src/circuit/eval.h src/gpu/image.h src/guest/q2b_map.h
	$(CXX) $(REFFLAGS) -o $@ src/tools/emit_gate.cc $(OUT)/tforward.o $(CIRC_OBJ) $(OUT)/image.o $(OUT)/sm_ref.o $(TEXT_OBJ)
model: $(OUT)/gpu_emit $(OUT)/gpu_run $(OUT)/emit_gate
.PHONY: model
