#include <dmlc/memory_io.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <unordered_map>
#include "/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/Deeplearning_Framework/TVM/tvm/src/runtime/library_module.h"
#include "nnvm/src/c_api/c_api_common.h"
namespace tvm {
namespace extract {
constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

// Module CreateModuleFromLibrary(ObjectPtr<Library> lib) {//参数为make_object<DSOLibrary>();
//   InitContextFunctions([lib](const char* fname) { return lib->GetSymbol(fname); });
//   auto n = make_object<LibraryModuleNode>(lib);
//   // Load the imported modules
//   const char* dev_mblob =
//       reinterpret_cast<const
//       char*>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));//保存参数的地方，通过符号表访问
//   Module root_mod;
//   if (dev_mblob != nullptr) {
//     root_mod = ProcessModuleBlob(dev_mblob, lib);
//   } else {
//     // Only have one single DSO Module
//     root_mod = Module(n);
//   }

//   // allow lookup of symbol from root (so all symbols are visible).
//   if (auto* ctx_addr = reinterpret_cast<void**>(lib->GetSymbol(runtime::symbol::tvm_module_ctx)))
//   {
//     *ctx_addr = root_mod.operator->();
//   }

//   return root_mod;
// }
inline runtime::NDArray Load(dmlc::Stream* strm) {
  uint64_t header, reserved;
  ICHECK(strm->Read(&header)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&reserved)) << "Invalid DLTensor file format";
  ICHECK(header == kTVMNDArrayMagic) << "Invalid DLTensor file format";
  DLContext ctx;
  int ndim;
  DLDataType dtype;
  ICHECK(strm->Read(&ctx)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&ndim)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&dtype)) << "Invalid DLTensor file format";
  ICHECK_EQ(ctx.device_type, kDLCPU) << "Invalid DLTensor context: can only save as CPU tensor";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    ICHECK(strm->ReadArray(&shape[0], ndim)) << "Invalid DLTensor file format";
  } 
  runtime::NDArray ret = runtime::NDArray::Empty(shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  ICHECK(strm->Read(&data_byte_size)) << "Invalid DLTensor file format";
  ICHECK(data_byte_size == num_elems * elem_bytes) << "Invalid DLTensor file format";
  auto read_ret = strm->Read(ret->data, data_byte_size);
  // Only check non-empty data
  if (ndim > 0 && shape[0] != 0) {
    ICHECK(read_ret) << "Invalid DLTensor file format";
  }
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(ret->data, elem_bytes, num_elems);
  }

  return ret;
}
runtime::Module GraphRuntimeFactoryModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string graph_json;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  std::string module_name;
  ICHECK(stream->Read(&graph_json));
  uint64_t sz;
  ICHECK(stream->Read(&sz));
  std::vector<std::string> names;
  ICHECK(stream->Read(&names));
  ICHECK(sz == names.size());
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::NDArray temp = Load(stream);
    params[names[i]] = temp;
  }
  ICHECK(stream->Read(&module_name));
  // auto exec = make_object<GraphRuntimeFactory>(graph_json, params, module_name);
  // return Module(exec);
}

runtime::Module ProcessModuleBlob(const char* mblob) {
  ICHECK(mblob != nullptr);
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = mblob[i];
    nbytes |= (c & 0xffUL) << (i * 8);
  }
  dmlc::MemoryFixedSizeStream fs(const_cast<char*>(mblob + sizeof(nbytes)),
                                 static_cast<size_t>(nbytes));
  dmlc::Stream* stream = &fs;
  uint64_t size;
  ICHECK(stream->Read(&size));
  std::vector<runtime::Module> modules;
  std::vector<uint64_t> import_tree_row_ptr;
  std::vector<uint64_t> import_tree_child_indices;
  for (uint64_t i = 0; i < size; ++i) {
    std::string tkey;
    ICHECK(stream->Read(&tkey));
    // Currently, _lib is for DSOModule, but we
    // don't have loadbinary function for it currently
    if (tkey == "_lib") {
      // auto dso_module = Module(make_object<LibraryModuleNode>(lib));
      // modules.emplace_back(dso_module);
    } else if (tkey == "_import_tree") {
      ICHECK(stream->Read(&import_tree_row_ptr));
      ICHECK(stream->Read(&import_tree_child_indices));
    } else {
      std::string loadkey = "runtime.module.loadbinary_";
      std::string fkey = loadkey + tkey;
      const runtime::PackedFunc* f = runtime::Registry::Get(fkey);
      if (f == nullptr) {
        std::string loaders = "";
        for (auto name : runtime::Registry::ListNames()) {
          if (name.rfind(loadkey, 0) == 0) {
            if (loaders.size() > 0) {
              loaders += ", ";
            }
            loaders += name.substr(loadkey.size());
          }
        }
        ICHECK(f != nullptr)
            << "Binary was created using " << tkey
            << " but a loader of that name is not registered. Available loaders are " << loaders
            << ". Perhaps you need to recompile with this runtime enabled.";
      }
      // runtime::Module m = (*f)(static_cast<void*>(stream));
      runtime::Module m = GraphRuntimeFactoryModuleLoadBinary(stream);
      //下一句报错-_-
      modules.emplace_back(m);
    }
  }
  //   // if we are using old dll, we don't have import tree
  //   // so that we can't reconstruct module relationship using import tree
  //   if (import_tree_row_ptr.empty()) {
  //     auto n = make_object<LibraryModuleNode>(lib);
  //     auto module_import_addr = ModuleInternal::GetImportsAddr(n.operator->());
  //     for (const auto& m : modules) {
  //       module_import_addr->emplace_back(m);
  //     }
  //     return Module(n);
  //   } else {
  //     for (size_t i = 0; i < modules.size(); ++i) {
  //       for (size_t j = import_tree_row_ptr[i]; j < import_tree_row_ptr[i + 1]; ++j) {
  //         auto module_import_addr = ModuleInternal::GetImportsAddr(modules[i].operator->());
  //         auto child_index = import_tree_child_indices[j];
  //         ICHECK(child_index < modules.size());
  //         module_import_addr->emplace_back(modules[child_index]);
  //       }
  //     }
  //   }
  //   ICHECK(!modules.empty());
  //   // invariance: root module is always at location 0.
  //   // The module order is collected via DFS
  return modules[0];
}
int ExtractJsonGraph(const char* mblob){
  tvm::extract::ProcessModuleBlob(mblob);
}
}  // namespace extract
}  // namespace tvm
int main(){
  std::string input = "apps/howto_deploy/lib/fromtf.txt";
  FILE* file = fopen(input.c_str(), "r");
  int len =  fseek(file,0L,SEEK_END);
  long lSize = ftell (file);
  rewind(file); 
   int num = lSize/sizeof(char);  
        char *pos = (char*) malloc (sizeof(char)*num);    
        if (pos == NULL)    
        {    
            printf("开辟空间出错");     
            return 0;   
        }   
        fread(pos,sizeof(char),num,file);  
       tvm::extract::ProcessModuleBlob(pos);
        free(pos);  
 
}