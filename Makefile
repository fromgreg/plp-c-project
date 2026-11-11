# cgrad build
#
#   make                  build everything in debug mode (default)
#   make release|sanitize build everything in that mode
#   make test             build + run the test runner (MODE=debug unless given)
#   make MODE=sanitize test
#   make leaks            run macOS `leaks` on the debug (non-ASan) test runner
#   make check            debug tests, sanitizer tests, leaks, XOR under ASan
#   make xor              build + run examples/xor.c in release mode
#   make clean

CC   := clang
MODE ?= debug
BUILD := build/$(MODE)

# ---- flags ------------------------------------------------------------------
CSTD     := -std=c11
WARN     := -Wall -Wextra -Wpedantic -Wshadow
CPPFLAGS := -Isrc
# Emit a .d file per object listing the headers it includes.
DEPFLAGS := -MMD -MP
LDLIBS   := -lm
SAN      := -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer

ifeq ($(MODE),debug)
  MODE_CFLAGS  := -O0 -g -Werror
  MODE_LDFLAGS :=
else ifeq ($(MODE),release)
  MODE_CFLAGS  := -O2 -DNDEBUG
  MODE_LDFLAGS :=
else ifeq ($(MODE),sanitize)
  # The sanitizer runtime must be linked in too, hence $(SAN) in both.
  MODE_CFLAGS  := -O1 -g -Werror $(SAN)
  MODE_LDFLAGS := $(SAN)
else
  $(error Unknown MODE '$(MODE)': use debug, release or sanitize)
endif

CFLAGS  := $(CSTD) $(WARN) $(MODE_CFLAGS)
LDFLAGS := $(MODE_LDFLAGS)

# Full stack traces for UBSan reports in every recipe that runs a binary.
export UBSAN_OPTIONS := print_stacktrace=1

# ---- sources and outputs ----------------------------------------------------
# src/*.c      -> library objects, linked into every binary
# examples/*.c -> one binary per file:  build/<mode>/<name>
# tests/*.c    -> one binary:           build/<mode>/test_runner
LIB_SRCS  := $(wildcard src/*.c)
EX_SRCS   := $(wildcard examples/*.c)
TEST_SRCS := $(wildcard tests/*.c)

LIB_OBJS  := $(LIB_SRCS:%.c=$(BUILD)/%.o)
EX_OBJS   := $(EX_SRCS:%.c=$(BUILD)/%.o)
TEST_OBJS := $(TEST_SRCS:%.c=$(BUILD)/%.o)
DEPS      := $(LIB_OBJS:.o=.d) $(EX_OBJS:.o=.d) $(TEST_OBJS:.o=.d)

EXAMPLES  := $(notdir $(EX_SRCS:.c=))
EX_BINS   := $(addprefix $(BUILD)/,$(EXAMPLES))
TEST_BIN  := $(BUILD)/test_runner

# ---- build ------------------------------------------------------------------
all: $(EX_BINS) $(TEST_BIN)

# Re-run make with a different MODE; each mode has its own build directory.
debug release sanitize:
	@$(MAKE) --no-print-directory MODE=$@ all

# Static pattern rule: only applies to the targets listed in $(EX_BINS).
# $@ = target, $^ = all prerequisites.
$(EX_BINS): $(BUILD)/%: $(BUILD)/examples/%.o $(LIB_OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(TEST_BIN): $(TEST_OBJS) $(LIB_OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Pattern rule: build/<mode>/dir/file.o from dir/file.c. $< = first prerequisite.
$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# Header dependencies from -MMD; '-' ignores them before the first build.
-include $(DEPS)

# ---- run --------------------------------------------------------------------
test: $(TEST_BIN)
	./$(TEST_BIN)

# `leaks` cannot inspect ASan binaries (ASan replaces malloc): use debug build.
leaks:
	@$(MAKE) --no-print-directory MODE=debug all
	leaks --atExit -- ./build/debug/test_runner

check:
	@$(MAKE) --no-print-directory MODE=debug test
	@$(MAKE) --no-print-directory MODE=sanitize test
	@$(MAKE) --no-print-directory leaks
	@$(MAKE) --no-print-directory MODE=sanitize all
	./build/sanitize/xor

# `make <example>` builds examples/<example>.c in release mode and runs it.
$(EXAMPLES):
	@$(MAKE) --no-print-directory MODE=release build/release/$@
	./build/release/$@

clean:
	rm -rf build

.PHONY: all debug release sanitize test leaks check clean $(EXAMPLES)
