import sys
from elftools.elf.elffile import ELFFile
"""
通过符号表找到compute_函数的地址(未完成)
"""
with open('./PReLU-shared_axes1.so', 'rb') as f:
    e = ELFFile(f)
    for section in e.iter_sections():
        print(hex(section['sh_addr']), section.name, section['sh_size'])
        if(section.name==".symtab"):            
            print("")
