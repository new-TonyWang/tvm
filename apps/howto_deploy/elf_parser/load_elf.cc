#include "./load_elf.h"
namespace tvm{
  namespace extract{
int fd;
int str_section_start;
int dynstr;
std::vector<Elf64_Phdr> phdrs;  //保存所有段头信息
std::vector<Elf64_Shdr> shdrs;  //保存所有节头信息
std::unordered_map<std::string, Elf64_Sym> syms_with_name;//保存所有symbol信息和name的对应
//找段在string section的位置
std::string searchNameInStringTable(int index) {
  lseek(fd, str_section_start + index, SEEK_SET);
  char tmp;
  int len = 0;
  while (read(fd, &tmp, 1) && tmp != char(0x00)) {
    len++;
  }
  char name[len + 1];
  name[len] = '\0';
  lseek(fd, str_section_start + index, SEEK_SET);
  read(fd, &name, len);
  lseek(fd, str_section_start, SEEK_SET);
  return name;
}

//找段在dystr section的位置
std::string searchNameIndystrTable(int index) {
  lseek(fd, dynstr + index, SEEK_SET);
  char tmp;
  int len = 0;
  while (read(fd, &tmp, 1) && tmp != char(0x00)) {
    len++;
  }
  char name[len + 1];
  name[len] = '\0';
  lseek(fd, dynstr + index, SEEK_SET);
  read(fd, &name, len);
  lseek(fd, dynstr, SEEK_SET);
  return name;
}

//找到String Section的位置
int findStrSection(std::vector<Elf64_Shdr> shdrs, Elf64_Ehdr ehdr) {
  return shdrs[ehdr.e_shstrndx].sh_offset;
}


void* ldopen(const char* __file) {
      printf("hookdlopen()");
  fd = open(__file, O_RDWR);
  Elf64_Ehdr ehdr;
  Elf64_Phdr phdr;
  Elf64_Shdr shdr;
 
  if (read(fd, &ehdr, sizeof(Elf64_Ehdr)) != sizeof(Elf64_Ehdr)) {
    puts("Read ELF header error");
  }

  lseek(fd, ehdr.e_phoff, SEEK_SET);
  for (int i = 0; i < ehdr.e_phnum; i++) {
    Elf64_Phdr tmp;
    if (read(fd, &tmp, sizeof(Elf64_Phdr)) != sizeof(Elf64_Phdr)) {
      puts("Read ELF header error");
    }
    phdrs.push_back(tmp);
  }

  lseek(fd, ehdr.e_shoff, SEEK_SET);
  for (int i = 0; i < ehdr.e_shnum; i++) {
    Elf64_Shdr tmp;
    if (read(fd, &tmp, sizeof(Elf64_Shdr)) != sizeof(Elf64_Shdr)) {
      puts("Read ELF header error");
    }
    shdrs.push_back(tmp);
  }
  //找strsection
  str_section_start = findStrSection(shdrs, ehdr);
  //保存带名字的section
  std::unordered_map<std::string, Elf64_Shdr> shdrs_with_name;
  for (int i = 0; i < ehdr.e_shnum; i++) {
    shdrs_with_name[searchNameInStringTable(shdrs[i].sh_name)] = shdrs[i];
  }

  //找dynsymtab
  Elf64_Shdr dynsym = shdrs_with_name[".dynsym"];
  //找dynstr
  dynstr = shdrs_with_name[".dynstr"].sh_offset;
  lseek(fd, dynsym.sh_offset, SEEK_SET);
  int dynsym_size = dynsym.sh_size / dynsym.sh_entsize;  //一共有几个symbol
  std::vector<Elf64_Sym> syms;                           //保存所有symbol信息
  //std::unordered_map<std::string, Elf64_Sym> syms_with_name; //保存所有symbol信息和name的对应
  for (int i = 0; i < dynsym_size; i++) {
    Elf64_Sym tmp;
    read(fd, &tmp, sizeof(Elf64_Sym));
    //syms_with_name[searchNameIndystrTable(tmp.st_name)]=tmp;
    syms.push_back(tmp);
  }
  for (int i = 0; i < dynsym_size; i++) {
    Elf64_Sym tmp;
    syms_with_name[searchNameIndystrTable(syms[i].st_name)]=syms[i];
    //syms.push_back(tmp);
  }
  
  return &fd;
}
// void* dlsym(void* __restrict __handle, const char* __restrict __name) {
//     //通过符号表找对应的数据
  
//     //return dlsym(__handle,__name);
//     return nullptr;
      
// }

char* readdata(std::string __name){
       printf("hookdlsym()");
   
        Elf64_Sym tmp = syms_with_name[__name];
        lseek(fd, syms_with_name[__name].st_value, SEEK_SET);
        char *name_data = (char*) malloc (sizeof(char)*(tmp.st_size+1));
        //char name_data[tmp.st_size+1];
        
       // mmap();
        read(fd,name_data,tmp.st_size);
        name_data[tmp.st_size] = '\0';
        return name_data;
    
}
char* load_elf(const char* elf_path){
   std::string elf_file = "apps/howto_deploy/lib/from_tensorflow_mod.so";
   std::string output = "/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/Deeplearning_Framework/TVM/tvm2/apps/howto_deploy/lib/mnist__02_llvm_x86.txt";
   ldopen(elf_path);
 
  
 std::string tvm_dev_mblob = "__tvm_dev_mblob";
  char* fd2 = readdata(tvm_dev_mblob);
  //const char* ad = reinterpret_cast<const char*>(dlsym(fd2,tvm_dev_mblob));
//   char* ad = readdata(tvm_dev_mblob);
  //  Elf64_Sym tmp = syms_with_name[tvm_dev_mblob];
  //       lseek(fd, syms_with_name[tvm_dev_mblob].st_value, SEEK_SET);
  //       //char name_data[tmp.st_size+1];
  //       char *name_data = (char*) malloc (sizeof(char)*(tmp.st_size+1));   
  //       name_data[tmp.st_size] = '\0';
  //      // mmap();
  //       read(fd,name_data,tmp.st_size);
int out = open(output.c_str(), O_RDWR | O_CREAT, 00777);
int len = syms_with_name[tvm_dev_mblob].st_size;
//lseek(out, len - 1, SEEK_END);
write(out, fd2, len);
close(fd);
return fd2;
}
}
}
// int main2(int argc, char** argv) {
//   std::string elf_file = "apps/howto_deploy/lib/from_tensorflow_mod.so";
//    std::string output = "apps/howto_deploy/lib/fromtf2.txt";
//   void* fd2 = ldopen(elf_file.c_str(), RTLD_LAZY | RTLD_LOCAL);
//  std::string tvm_dev_mblob = "__tvm_dev_mblob";
//   //const char* ad = reinterpret_cast<const char*>(dlsym(fd2,tvm_dev_mblob));
//    char* ad = readdata(tvm_dev_mblob);
//    Elf64_Sym tmp = syms_with_name[tvm_dev_mblob];
//         lseek(fd, syms_with_name[tvm_dev_mblob].st_value, SEEK_SET);
//       //   //char name_data[tmp.st_size+1];
//       //   char *name_data = (char*) malloc (sizeof(char)*(tmp.st_size+1));   
//       //   name_data[tmp.st_size] = '\0';
//       //  // mmap();
//         read(fd,ad,tmp.st_size);
// int out = open(output.c_str(), O_RDWR | O_CREAT, 00777);
// int len = syms_with_name[tvm_dev_mblob].st_size;
// //lseek(out, len - 1, SEEK_END);
// write(out, ad, len);
// //   char* outputs = (char*)mmap(NULL,len, PROT_READ | PROT_WRITE, MAP_SHARED,out,0);
// //    memcpy(outputs, ad, len);
// //    munmap(outputs,len);

//   close(out);
//   close(fd);
//   free(ad);
//   return 0;
// }
