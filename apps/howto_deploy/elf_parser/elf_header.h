#define _LINUX64
#if defined(_ANDROID64)
    /* 64-bit ELF base types. */
typedef __u64	Elf64_Addr;
typedef __u16	Elf64_Half;
typedef __s16	Elf64_SHalf;
typedef __u64	Elf64_Off;
typedef __s32	Elf64_Sword;
typedef __u32	Elf64_Word;
typedef __u64	Elf64_Xword;
typedef __s64	Elf64_Sxword;
#define EI_NIDENT	16
typedef struct elf64_hdr {
  unsigned char	e_ident[EI_NIDENT];	/* ELF "magic number" */
  Elf64_Half e_type;
  Elf64_Half e_machine;
  Elf64_Word e_version;
  Elf64_Addr e_entry;		/* Entry point virtual address */
  Elf64_Off e_phoff;		/* Program header table file offset 
                        程序头在文件的偏移地址，可以用该偏移定位程序头的开始位置*/
  Elf64_Off e_shoff;		/* Section header table file offset 段头内容在这个文件的偏移值*/
  Elf64_Word e_flags;
  Elf64_Half e_ehsize;
  Elf64_Half e_phentsize;
  Elf64_Half e_phnum;      /*程序头的个数*/
  Elf64_Half e_shentsize;
  Elf64_Half e_shnum;       /*段头的个数*/
  Elf64_Half e_shstrndx;    /*String在整个段表中的索引值*/
} Elf64_Ehdr;
#endif
#if defined(_ANDROID32)
#endif
#if defined(_LINUX64)
#include <elf.h>
#endif

#if defined(_LINUX32)
#include <elf.h>
#endif