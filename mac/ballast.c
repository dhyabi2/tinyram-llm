// ballast.c : cc -O2 -o ballast ballast.c ; usage: sudo ./ballast <bytes>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
int main(int c,char**v){size_t n=strtoull(v[1],0,10);
 char*p=mmap(0,n,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
 if(p==MAP_FAILED){perror("mmap");return 1;}
 for(size_t i=0;i<n;i+=4096)p[i]=1; /* touch */
 if(mlock(p,n)){perror("mlock");return 2;}
 printf("WIRED %zu bytes pid %d\n",n,getpid());fflush(stdout);
 pause();return 0;}
