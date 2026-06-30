// Context-create limit probe. Loads each bin given on argv via contextCreateFromBinary,
// prints OK/FAIL + the QNN error code + cumulative resident MB. A trailing "free" arg frees
// each context before the next (isolates per-binary size limit from cumulative-residency limit).
// Usage: ctx_probe <aotdir> <bin1> <bin2> ... [free]
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTypes.h"
#include "QnnLog.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"

static QNN_INTERFACE_VER_TYPE q;
static std::vector<uint8_t> rd(const char* p){ FILE*f=fopen(p,"rb"); if(!f){perror(p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<uint8_t> b(n); if(fread(b.data(),1,n,f)!=(size_t)n)exit(1); fclose(f); return b; }

int main(int argc, char** argv){
  if (argc < 3){ fprintf(stderr,"usage: %s <aotdir> <bin1> [bin2 ...] [free]\n", argv[0]); return 2; }
  bool freeEach = (strcmp(argv[argc-1],"free")==0);
  int lastBin = freeEach ? argc-1 : argc;
  std::string ld = argv[1];
  void* lib = dlopen((ld+"/libQnnHtp.so").c_str(), RTLD_NOW|RTLD_GLOBAL); if(!lib){fprintf(stderr,"%s\n",dlerror());return 1;}
  const QnnInterface_t** prov=nullptr; uint32_t np=0;
  ((Qnn_ErrorHandle_t(*)(const QnnInterface_t***,uint32_t*))dlsym(lib,"QnnInterface_getProviders"))(&prov,&np);
  q = prov[0]->QNN_INTERFACE_VER_NAME;
  Qnn_LogHandle_t log=nullptr; q.logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &log);
  Qnn_BackendHandle_t be=nullptr; if(q.backendCreate(log,nullptr,&be)){fprintf(stderr,"backendCreate FAIL\n");return 1;}
  Qnn_DeviceHandle_t dev=nullptr; q.deviceCreate(log,nullptr,&dev);
  double cumMB=0;
  std::vector<Qnn_ContextHandle_t> held;
  for (int i=2;i<lastBin;i++){
    auto bin = rd(argv[i]);
    double mb = bin.size()/1048576.0;
    Qnn_ContextHandle_t ctx=nullptr;
    Qnn_ErrorHandle_t e = q.contextCreateFromBinary(be,dev,nullptr,bin.data(),bin.size(),&ctx,nullptr);
    if (e){ printf("FAIL  %-28s  %.0f MB  err=0x%llx  (cum before=%.0f MB)\n", argv[i], mb, (unsigned long long)e, cumMB); }
    else  { cumMB += freeEach?0:mb; printf("OK    %-28s  %.0f MB  cum_resident=%.0f MB\n", argv[i], mb, cumMB); if(freeEach) q.contextFree(ctx,nullptr); else held.push_back(ctx); }
  }
  for (auto c: held) q.contextFree(c,nullptr);
  return 0;
}
