#include <dlfcn.h>
#include <dmlc/memory_io.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../src/runtime/library_module.h"
#include "../../../src/runtime/runtime_base.h"
#include "./elf_header.h"

namespace tvm {
namespace extract {
std::string searchNameInStringTable(int index);
std::string searchNameIndystrTable(int index);
int findStrSection(std::vector<Elf64_Shdr> shdrs, Elf64_Ehdr ehdr);
char* load_elf(const char* elf_path);
}  // namespace extract
}  // namespace tvm