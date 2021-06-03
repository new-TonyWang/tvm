
#include "./load_elf.h"
namespace tvm {
namespace extract {

class Extracter{
public:
   uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;
    std::string graph_json;
    std::string module_name;
    uint64_t sz;
    dmlc::Stream* stream;

 runtime::NDArray Load() {
    uint64_t header, reserved;
  ICHECK(stream->Read(&header)) << "Invalid DLTensor file format";
  ICHECK(stream->Read(&reserved)) << "Invalid DLTensor file format";
  ICHECK(header == kTVMNDArrayMagic) << "Invalid DLTensor file format";
  Device dev;
  int ndim;
  DLDataType dtype;
  ICHECK(stream->Read(&dev)) << "Invalid DLTensor file format";
  ICHECK(stream->Read(&ndim)) << "Invalid DLTensor file format";
  ICHECK(stream->Read(&dtype)) << "Invalid DLTensor file format";
  ICHECK_EQ(dev.device_type, kDLCPU) << "Invalid DLTensor device: can only save as CPU tensor";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    ICHECK(stream->ReadArray(&shape[0], ndim)) << "Invalid DLTensor file format";
  }
  tvm::runtime::NDArray ret = tvm::runtime::NDArray::Empty(shape, dtype, dev);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  ICHECK(stream->Read(&data_byte_size)) << "Invalid DLTensor file format";
  ICHECK(data_byte_size == num_elems * elem_bytes) << "Invalid DLTensor file format";
  auto read_ret = stream->Read(ret->data, data_byte_size);
  // Only check non-empty data
  if (ndim > 0 && shape[0] != 0) {
    ICHECK(read_ret) << "Invalid DLTensor file format";
  }
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(ret->data, elem_bytes, num_elems);
  }
  
  return ret;
}
void* GraphRuntimeFactoryModuleLoadBinary() {
  //dmlc::Stream* stream = static_cast<dmlc::Stream*>(stream);
  //std::string graph_json;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  //std::string module_name;
  ICHECK(stream->Read(&graph_json));
  ICHECK(stream->Read(&sz));
  std::vector<std::string> names;
  ICHECK(stream->Read(&names));
  ICHECK(sz == names.size());
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::NDArray temp = Load();
    
    params[names[i]] = temp;
  }
  ICHECK(stream->Read(&module_name));
  // auto exec = make_object<GraphRuntimeFactory>(graph_json, params, module_name);
  // return Module(exec);
}

void ProcessModuleBlob(const char* mblob) {
  ICHECK(mblob != nullptr);
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = mblob[i];
    nbytes |= (c & 0xffUL) << (i * 8);
  }
  dmlc::MemoryFixedSizeStream fs(const_cast<char*>(mblob + sizeof(nbytes)),
                                 static_cast<size_t>(nbytes));
  stream = &fs;
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
      // std::string loadkey = "runtime.module.loadbinary_";
      // std::string fkey = loadkey + tkey;
      // const runtime::PackedFunc* f = runtime::Registry::Get(fkey);
      // if (f == nullptr) {
      //   std::string loaders = "";
      //   for (auto name : runtime::Registry::ListNames()) {
      //     if (name.rfind(loadkey, 0) == 0) {
      //       if (loaders.size() > 0) {
      //         loaders += ", ";
      //       }
      //       loaders += name.substr(loadkey.size());
      //     }
      //   }
      //   ICHECK(f != nullptr)
      //       << "Binary was created using " << tkey
      //       << " but a loader of that name is not registered. Available loaders are " << loaders
      //       << ". Perhaps you need to recompile with this runtime enabled.";
      // }
      // runtime::Module m = (*f)(static_cast<void*>(stream));
      GraphRuntimeFactoryModuleLoadBinary();
      //下一句报错-_-
      //modules.emplace_back(m);
    }
  }

}

};
// int TVMFuncCall(TVMFunctionHandle func, TVMValue* args, int* arg_type_codes, int num_args,
//                 TVMValue* ret_val, int* ret_type_code) {
  

//   TVMRetValue rv;
//   (*static_cast<const PackedFunc*>(func)).CallPacked(TVMArgs(args, arg_type_codes, num_args), &rv);
//   // handle return string.
//   if (rv.type_code() == kTVMStr || rv.type_code() == kTVMDataType || rv.type_code() == kTVMBytes) {
//     TVMRuntimeEntry* e = TVMAPIRuntimeStore::Get();
//     if (rv.type_code() != kTVMDataType) {
//       e->ret_str = *rv.ptr<std::string>();
//     } else {
//       e->ret_str = rv.operator std::string();
//     }
//     if (rv.type_code() == kTVMBytes) {
//       e->ret_bytes.data = e->ret_str.c_str();
//       e->ret_bytes.size = e->ret_str.length();
//       *ret_type_code = kTVMBytes;
//       ret_val->v_handle = &(e->ret_bytes);
//     } else {
//       *ret_type_code = kTVMStr;
//       ret_val->v_str = e->ret_str.c_str();
//     }
//   } else {
//     rv.MoveToCHost(ret_val, ret_type_code);
//   }
  
// }



}  // namespace extract
}  // namespace tvm
//extern "C"{
int  readNextNDarray(tvm::extract::Extracter* obj, TVMValue* ret_val, int* ret_type_code){
    API_BEGIN();
    tvm::runtime::TVMRetValue rv;
    tvm::runtime::NDArray temp = obj->Load();
    rv = temp;
    rv.MoveToCHost(ret_val, ret_type_code);
    API_END();
    
}
  
 tvm::extract::Extracter* ExtractJsonGraph(const char* mblob){
    tvm::extract::Extracter* extracter = new tvm::extract::Extracter;
    extracter->ProcessModuleBlob(mblob);
    return extracter;
}
void Clean(tvm::extract::Extracter* obj,char* ptr){
  free(obj);
  free(ptr);
}
//}
int main(){
  // std::string input = "apps/howto_deploy/lib/fromtf.txt";
  // FILE* file = fopen(input.c_str(), "r");
  // int len =  fseek(file,0L,SEEK_END);
  // long lSize = ftell (file);
  // rewind(file); 
  //  int num = lSize/sizeof(char);  
  //       char *pos = (char*) malloc (sizeof(char)*num);    
  //       if (pos == NULL)    
  //       {    
  //           printf("开辟空间出错");     
  //           return 0;   
  //       }   
  //       fread(pos,sizeof(char),num,file);  
  //      tvm::extract::ProcessModuleBlob(pos);
  //       free(pos);  
  std::string name = "/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/Deeplearning_Framework/TVM/tvm2/apps/howto_deploy/lib/mnist_02_llvm_x86.so";
  const char* prt = tvm::extract::load_elf(name.c_str());
  tvm::extract::Extracter* a =  ExtractJsonGraph(prt);
  return 0;
}