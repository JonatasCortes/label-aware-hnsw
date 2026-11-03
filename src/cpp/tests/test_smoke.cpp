// Smoke test: proves that the CMake build and the ctest runner work,
// and that the compiler is actually using C++20.

#include <cstdio>

int main() {
#if __cplusplus < 202002L
    std::fprintf(stderr, "test_smoke: expected C++20 or newer, got __cplusplus=%ld\n",
                 static_cast<long>(__cplusplus));
    return 1;
#else
    std::puts("test_smoke: OK");
    return 0;
#endif
}