// Persistent-context in-memory chain runner for the 512² banked Bonsai DiT (the 0.933 set).
// resident=0/1 load the 27 separate LN-fixed V79 context binaries (pro + 5 dbl + 20 sgl + epi) —
// the working adb path. resident>=2 is the MERGED path (Rq1 = pro+5dbl+epi resident; the singles
// in 3 contexts sglA/B/Cq1; resident-2 = how many singles ctx kept resident) — written to cut the
// in-app FastRPC create-leak, but it is a DOCUMENTED DEAD END: the V79 refuses to create a context
// holding more than ~3 fp16 single-blocks (err 0x3ea; NOT a size cap — see runner/ctx_probe.cpp +
// the repo README), so the 7/7/6 singles contexts fail contextCreateFromBinary. Kept for the record.
// All modes force the
// DSP power corner (DCVS off + TURBO_PLUS), and runs the full FLUX 4-step rectified-flow
// rollout in RAM:  pro -> 5x double -> cat[txt,img] -> 20x single -> slice img -> epi -> vel
//                  x += (sig[i+1]-sig[i]) * vel    (Euler)
// Activations pass in RAM as float; qIn/dqOut requantise at each graph boundary to the bin's
// declared dtype (FLOAT_16 for pro+singles, SFIXED/UFIXED_16 for doubles+epi) — matching the
// host-orchestrated q1_chain_run.py exactly. Mac does patchify(init)/unpatchify(final).
// Reads <io>/latent.raw (patched [1024,128]), context.raw, cos.raw, sin.raw; writes final_chain.raw.
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>

#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnLog.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

static QNN_INTERFACE_VER_TYPE q;
static QNN_SYSTEM_INTERFACE_VER_TYPE s;

static double now_ms() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static std::vector<uint8_t> rd(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(n); if (fread(b.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return b;
}
static std::vector<float> rdf(const char* p) { auto b = rd(p); std::vector<float> v(b.size()/4); memcpy(v.data(), b.data(), b.size()); return v; }
static size_t esz(Qnn_DataType_t t) {
  switch (t) { case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_UINT_32: case QNN_DATATYPE_INT_32: return 4;
    case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_INT_16: return 2; default: return 1; } }
static size_t nel(const Qnn_Tensor_t& t) { size_t n = 1; for (uint32_t i = 0; i < t.v2.rank; i++) n *= t.v2.dimensions[i]; return n; }

struct Graph {
  Qnn_ContextHandle_t ctx = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  std::string name;
  std::vector<Qnn_Tensor_t> ins, outs;
  std::vector<std::vector<uint8_t>> ib, ob;
  std::vector<int> argPos, outPos;
};

static int idxFromName(const char* nm, const char* tag) {
  const char* p = strstr(nm, tag); if (!p) return -1; int v = -1; sscanf(p + strlen(tag), "%d", &v); return v;
}

// Fill a Graph's input/output tensors + arg/out position maps from its graph-info.
// Shared by loadGraph (one ctx = one graph) and loadCtx (one ctx = many graphs).
static void setupGraphIO(Graph& G, const QnnSystemContext_GraphInfoV1_t* g) {
  G.name = g->graphName;
  uint32_t nIn = g->numGraphInputs, nOut = g->numGraphOutputs;
  G.ins.assign(g->graphInputs, g->graphInputs + nIn);
  G.outs.assign(g->graphOutputs, g->graphOutputs + nOut);
  G.ib.resize(nIn); G.ob.resize(nOut);
  G.argPos.assign(nIn, -1); G.outPos.assign(nOut, -1);
  for (uint32_t i = 0; i < nIn; i++) {
    size_t b = nel(G.ins[i]) * esz(G.ins[i].v2.dataType); G.ib[i].assign(b, 0);
    G.ins[i].v2.memType = QNN_TENSORMEMTYPE_RAW; G.ins[i].v2.clientBuf.data = G.ib[i].data(); G.ins[i].v2.clientBuf.dataSize = (uint32_t)b;
    int k = idxFromName(G.ins[i].v2.name, "args_"); if (k >= 0 && k < (int)nIn) G.argPos[k] = i;
  }
  for (uint32_t i = 0; i < nOut; i++) {
    size_t b = nel(G.outs[i]) * esz(G.outs[i].v2.dataType); G.ob[i].assign(b, 0);
    G.outs[i].v2.memType = QNN_TENSORMEMTYPE_RAW; G.outs[i].v2.clientBuf.data = G.ob[i].data(); G.outs[i].v2.clientBuf.dataSize = (uint32_t)b;
    int k = idxFromName(G.outs[i].v2.name, "output_"); if (k >= 0 && k < (int)nOut) G.outPos[k] = i;
  }
}

// One context binary -> one graph (separate-bins path: 27 files, one graph each).
static Graph loadGraph(Qnn_BackendHandle_t be, Qnn_DeviceHandle_t dev, const char* path) {
  fprintf(stderr, "[load.read %s]\n", path);
  Graph G; auto bin = rd(path);
  QnnSystemContext_Handle_t sc = nullptr; s.systemContextCreate(&sc);
  const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bis = 0;
  if (s.systemContextGetBinaryInfo(sc, bin.data(), bin.size(), &bi, &bis)) { fprintf(stderr, "binInfo %s\n", path); exit(1); }
  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) graphs = bi->contextBinaryInfoV3.graphs;
  else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) graphs = bi->contextBinaryInfoV2.graphs;
  else graphs = bi->contextBinaryInfoV1.graphs;
  auto* g = (const QnnSystemContext_GraphInfoV1_t*)&graphs[0].graphInfoV1;
  fprintf(stderr, "[load.ctxcreate %s %zuMB]\n", path, bin.size() >> 20);
  if (q.contextCreateFromBinary(be, dev, nullptr, bin.data(), bin.size(), &G.ctx, nullptr)) { fprintf(stderr, "ctx %s\n", path); exit(1); }
  if (q.graphRetrieve(G.ctx, g->graphName, &G.graph)) { fprintf(stderr, "retrieve %s\n", path); exit(1); }
  fprintf(stderr, "[load.ok %s]\n", path);
  setupGraphIO(G, g);
  return G;
}

// quantise float src -> graph input arg k (FLOAT_16 / FLOAT_32 / per-tensor SFIXED/UFIXED_16)
static void qIn(Graph& G, int k, const float* src) {
  Qnn_Tensor_t& t = G.ins[G.argPos[k]]; size_t n = nel(t);
  Qnn_DataType_t dt = t.v2.dataType;
  if (dt == QNN_DATATYPE_FLOAT_32) { memcpy(t.v2.clientBuf.data, src, n * 4); return; }
  if (dt == QNN_DATATYPE_FLOAT_16) { __fp16* d = (__fp16*)t.v2.clientBuf.data; for (size_t i = 0; i < n; i++) d[i] = (__fp16)src[i]; return; }
  float scale = t.v2.quantizeParams.scaleOffsetEncoding.scale; int off = t.v2.quantizeParams.scaleOffsetEncoding.offset;
  bool sgnd = dt == QNN_DATATYPE_SFIXED_POINT_16;
  int lo = sgnd ? -32768 : 0, hi = sgnd ? 32767 : 65535;
  if (sgnd) { int16_t* d = (int16_t*)t.v2.clientBuf.data; for (size_t i = 0; i < n; i++) { int v = (int)lroundf(src[i] / scale) - off; d[i] = v < lo ? lo : v > hi ? hi : v; } }
  else { uint16_t* d = (uint16_t*)t.v2.clientBuf.data; for (size_t i = 0; i < n; i++) { int v = (int)lroundf(src[i] / scale) - off; d[i] = v < lo ? lo : v > hi ? hi : v; } }
}
// dequant graph output k -> float dst
static void dqOut(Graph& G, int k, float* dst) {
  Qnn_Tensor_t& t = G.outs[G.outPos[k]]; size_t n = nel(t);
  Qnn_DataType_t dt = t.v2.dataType;
  if (dt == QNN_DATATYPE_FLOAT_32) { memcpy(dst, t.v2.clientBuf.data, n * 4); return; }
  if (dt == QNN_DATATYPE_FLOAT_16) { __fp16* d = (__fp16*)t.v2.clientBuf.data; for (size_t i = 0; i < n; i++) dst[i] = (float)d[i]; return; }
  float scale = t.v2.quantizeParams.scaleOffsetEncoding.scale; int off = t.v2.quantizeParams.scaleOffsetEncoding.offset;
  if (dt == QNN_DATATYPE_SFIXED_POINT_16) { int16_t* d = (int16_t*)t.v2.clientBuf.data; for (size_t i = 0; i < n; i++) dst[i] = (d[i] + off) * scale; }
  else { uint16_t* d = (uint16_t*)t.v2.clientBuf.data; for (size_t i = 0; i < n; i++) dst[i] = ((int)d[i] + off) * scale; }
}
static void execG(Graph& G) { fprintf(stderr, "[exec %s]\n", G.name.c_str()); if (q.graphExecute(G.graph, G.ins.data(), G.ins.size(), G.outs.data(), G.outs.size(), nullptr, nullptr)) { fprintf(stderr, "exec %s fail\n", G.name.c_str()); exit(1); } fprintf(stderr, "[exec.ok %s]\n", G.name.c_str()); }
static void freeGraph(Graph& G) { if (G.ctx) q.contextFree(G.ctx, nullptr); G.ctx = nullptr; }

// One context binary -> MANY graphs (merged path: 3 files holding 7/10/10 graphs).
// Folding the 27 separate bins into 3 contexts cuts contextCreateFromBinary calls from
// ~67 over a 3-step run to ~7, under the untrusted_app FastRPC dspqueue leak limit (~10)
// that stalls the in-app (tappable) chain. Graphs share one ctx; freeCtx frees it once.
struct Ctx {
  Qnn_ContextHandle_t ctx = nullptr;
  std::vector<Graph> graphs;
  Graph* byName(const std::string& n) {
    for (auto& g : graphs) if (g.name == n) return &g;
    fprintf(stderr, "graph '%s' not found in context (have:", n.c_str());
    for (auto& g : graphs) fprintf(stderr, " %s", g.name.c_str());
    fprintf(stderr, ")\n"); exit(1);
  }
};
static Ctx loadCtx(Qnn_BackendHandle_t be, Qnn_DeviceHandle_t dev, const char* path) {
  fprintf(stderr, "[loadctx.read %s]\n", path);
  Ctx C; auto bin = rd(path);
  QnnSystemContext_Handle_t sc = nullptr; s.systemContextCreate(&sc);
  const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bis = 0;
  if (s.systemContextGetBinaryInfo(sc, bin.data(), bin.size(), &bi, &bis)) { fprintf(stderr, "binInfo %s\n", path); exit(1); }
  const QnnSystemContext_GraphInfo_t* graphs = nullptr; uint32_t numGraphs = 0;
  if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) { graphs = bi->contextBinaryInfoV3.graphs; numGraphs = bi->contextBinaryInfoV3.numGraphs; }
  else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { graphs = bi->contextBinaryInfoV2.graphs; numGraphs = bi->contextBinaryInfoV2.numGraphs; }
  else { graphs = bi->contextBinaryInfoV1.graphs; numGraphs = bi->contextBinaryInfoV1.numGraphs; }
  fprintf(stderr, "[loadctx.ctxcreate %s %zuMB %u graphs]\n", path, bin.size() >> 20, numGraphs);
  if (q.contextCreateFromBinary(be, dev, nullptr, bin.data(), bin.size(), &C.ctx, nullptr)) { fprintf(stderr, "ctx %s\n", path); exit(1); }
  C.graphs.resize(numGraphs);
  for (uint32_t gi = 0; gi < numGraphs; gi++) {
    auto* g = (const QnnSystemContext_GraphInfoV1_t*)&graphs[gi].graphInfoV1;
    Graph& G = C.graphs[gi]; G.ctx = nullptr;   // ctx owned by Ctx, not the per-graph view
    if (q.graphRetrieve(C.ctx, g->graphName, &G.graph)) { fprintf(stderr, "retrieve %s in %s\n", g->graphName, path); exit(1); }
    setupGraphIO(G, g);
  }
  fprintf(stderr, "[loadctx.ok %s: %u graphs]\n", path, numGraphs);
  return C;
}
static void freeCtx(Ctx& C) { if (C.ctx) q.contextFree(C.ctx, nullptr); C.ctx = nullptr; C.graphs.clear(); }

// FLUX schedule (matches q1_chain_run.flux_sigmas(n, 1024))
static void flux_sigmas(int n, double L, std::vector<double>& out) {
  double a1=8.73809524e-05, b1=1.89833333, a2=0.00016927, b2=0.45666666, mu;
  if (L > 4300) mu = a2*L + b2;
  else { double m2=a2*L+b2, m1=a1*L+b1, a=(m2-m1)/190.0; mu = a*n + (m2 - 200.0*a); }
  double em = exp(mu);
  out.clear();
  for (int i=0;i<n;i++) out.push_back(em / (em + (1.0/(1.0-(1.0-1.0/n)*i/(n-1)) - 1.0)));
  out.push_back(0.0);
}
static void temb_of(double sg, float* out) {     // [256] = concat(cos(a), sin(a)), a[i]=sg*1000*exp(-ln(1e4)*i/128)
  double t = sg * 1000.0;
  for (int i=0;i<128;i++){ double f=exp(-log(10000.0)*i/128.0), a=t*f; out[i]=(float)cos(a); out[128+i]=(float)sin(a); }
}

static int run_chain(const char* binsDir, const char* ioDir, int steps, int resident) {
  auto BP = [&](const std::string& f){ return std::string(binsDir) + "/" + f; };
  auto IP = [&](const std::string& f){ return std::string(ioDir) + "/" + f; };

  void* lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL); if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  const QnnInterface_t** prov = nullptr; uint32_t np = 0;
  ((Qnn_ErrorHandle_t(*)(const QnnInterface_t***, uint32_t*))dlsym(lib, "QnnInterface_getProviders"))(&prov, &np);
  q = prov[0]->QNN_INTERFACE_VER_NAME;
  void* sysl = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL); if (!sysl) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  const QnnSystemInterface_t** sp = nullptr; uint32_t nsp = 0;
  ((Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t***, uint32_t*))dlsym(sysl, "QnnSystemInterface_getProviders"))(&sp, &nsp);
  s = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

  Qnn_LogHandle_t log = nullptr; q.logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &log);
  Qnn_BackendHandle_t be = nullptr; if (q.backendCreate(log, nullptr, &be)) { fprintf(stderr, "backendCreate\n"); return 1; }
  // Device. Default = signed PD (nullptr config). Set QNN_UNSIGNED_PD=1 to request
  // unsigned PD (some app contexts need it to load an unsigned skel from app dirs).
  Qnn_DeviceHandle_t dev = nullptr;
  if (getenv("QNN_UNSIGNED_PD")) {
    QnnHtpDevice_CustomConfig_t pdCustom; memset(&pdCustom, 0, sizeof(pdCustom));
    pdCustom.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
    pdCustom.useSignedProcessDomain.deviceId = 0;
    pdCustom.useSignedProcessDomain.useSignedProcessDomain = false;
    QnnDevice_Config_t pdDevCfg; memset(&pdDevCfg, 0, sizeof(pdDevCfg));
    pdDevCfg.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM; pdDevCfg.customConfig = &pdCustom;
    const QnnDevice_Config_t* devCfgs[] = { &pdDevCfg, nullptr };
    if (q.deviceCreate(log, devCfgs, &dev)) { fprintf(stderr, "deviceCreate(unsignedPD) failed\n"); }
    else fprintf(stderr, "[pd] unsigned\n");
  } else {
    q.deviceCreate(log, nullptr, &dev);
  }

  QnnDevice_Infrastructure_t infra = nullptr;
  if (!q.deviceGetInfrastructure(&infra) && infra) {
    auto* hi = (QnnHtpDevice_Infrastructure_t*)infra; auto pf = hi->perfInfra; uint32_t pid = 0; pf.createPowerConfigId(0, 0, &pid);
    QnnHtpPerfInfrastructure_PowerConfig_t c; memset(&c, 0, sizeof(c));
    c.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3; c.dcvsV3Config.contextId = pid;
    c.dcvsV3Config.setDcvsEnable = 1; c.dcvsV3Config.dcvsEnable = 0;
    c.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    c.dcvsV3Config.setSleepLatency = 1; c.dcvsV3Config.sleepLatency = 40;
    c.dcvsV3Config.setSleepDisable = 1; c.dcvsV3Config.sleepDisable = 1;
    c.dcvsV3Config.setBusParams = 1; c.dcvsV3Config.busVoltageCornerMin = c.dcvsV3Config.busVoltageCornerTarget = c.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    c.dcvsV3Config.setCoreParams = 1; c.dcvsV3Config.coreVoltageCornerMin = c.dcvsV3Config.coreVoltageCornerTarget = c.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    const QnnHtpPerfInfrastructure_PowerConfig_t* arr[] = { &c, nullptr };
    fprintf(stderr, "[power] TURBO_PLUS rc=%llu\n", (unsigned long long)pf.setPowerConfig(pid, arr));
  }

  // ===== MERGED-CONTEXT PATH (resident>=2): the tappable / in-app fix =====
  // A single QNN context binary can't exceed ~2 GiB (contextCreateFromBinary err 0x3ea),
  // so the 20 singles are packed into 3 contexts of <=8 (sglA=sgl0-6, sglB=sgl7-13,
  // sglC=sgl14-19, ~1.7/1.7/1.4 GiB). R (pro+5dbl+epi, 1.53 GiB) is always resident.
  // The first (resident-2) singles contexts are ALSO kept resident (created once); the
  // rest are streamed per step. Total contextCreateFromBinary calls = (1 + residentSgl)
  // + (3 - residentSgl) * steps, kept under the ~10 untrusted_app FastRPC leak limit:
  //   resident=3 (1 resident sgl): 3 steps -> 8 creates, ~4.9 GB peak  (robust target)
  //   resident=4 (2 resident sgl): 4 steps -> 7 creates, ~6.3 GB peak  (near reboot ceiling)
  //   resident=2 (0 resident sgl): 3 steps -> 10 creates, ~3.3 GB peak (at the wall)
  if (resident >= 2) {
    int residentSgl = resident - 2; if (residentSgl > 3) residentSgl = 3;
    const int D = 3072, IMG = 1024, TXT = 512, S = 1536;
    auto x = rdf(IP("latent.raw").c_str());
    auto context = rdf(IP("context.raw").c_str());
    auto cos = rdf(IP("cos.raw").c_str()), sin = rdf(IP("sin.raw").c_str());
    std::vector<float> img(IMG*D), txt(TXT*D), imgmod(6*D), txtmod(6*D), smod(3*D), sv(D),
                       xm(S*D), imgtok(IMG*D), vel(IMG*128), temb(256);
    std::vector<double> SIG; flux_sigmas(steps, 1024.0, SIG);
    fprintf(stderr, "[sched] sigmas %.4f %.4f %.4f %.4f %.4f\n", SIG[0],SIG[1],SIG[2],SIG[3],SIG[4]);

    // singles split into 3 contexts (<2GiB each); inclusive index ranges
    struct SglGroup { const char* path; int first, last; };
    SglGroup groups[3] = { {"sglAq1.bin",0,6}, {"sglBq1.bin",7,13}, {"sglCq1.bin",14,19} };

    fprintf(stderr, "[merged] R resident + %d/3 singles ctx resident, %d streamed/step (%d creates over %d steps)\n",
            residentSgl, 3-residentSgl, (1+residentSgl) + (3-residentSgl)*steps, steps);
    Ctx Rc = loadCtx(be, dev, BP("Rq1.bin").c_str());
    Graph* pro = Rc.byName("prob"); Graph* epi = Rc.byName("epib");
    Graph* dbl[5]; for (int j = 0; j < 5; j++) dbl[j] = Rc.byName("dbl" + std::to_string(j) + "b");
    Ctx resCtx[3]; bool isRes[3] = { false, false, false };       // resident singles contexts (created once)
    for (int gi = 0; gi < residentSgl; gi++) { resCtx[gi] = loadCtx(be, dev, BP(groups[gi].path).c_str()); isRes[gi] = true; }

    auto runSingles = [&](Ctx& c, int first, int last) {
      for (int j = first; j <= last; j++) { Graph* g = c.byName("sgl" + std::to_string(j) + "b");
        qIn(*g, 0, xm.data()); qIn(*g, 1, cos.data()); qIn(*g, 2, sin.data()); qIn(*g, 3, smod.data());
        execG(*g); dqOut(*g, 0, xm.data()); }
    };

    double tc0 = now_ms();
    for (int step = 0; step < steps; step++) {
      double ts = now_ms();
      temb_of(SIG[step], temb.data());
      qIn(*pro, 0, x.data()); qIn(*pro, 1, context.data()); qIn(*pro, 2, temb.data());
      execG(*pro);
      dqOut(*pro, 0, img.data()); dqOut(*pro, 1, txt.data()); dqOut(*pro, 2, imgmod.data());
      dqOut(*pro, 3, txtmod.data()); dqOut(*pro, 4, smod.data()); dqOut(*pro, 5, sv.data());
      for (int j = 0; j < 5; j++) {
        qIn(*dbl[j], 0, img.data()); qIn(*dbl[j], 1, txt.data()); qIn(*dbl[j], 2, cos.data());
        qIn(*dbl[j], 3, sin.data()); qIn(*dbl[j], 4, imgmod.data()); qIn(*dbl[j], 5, txtmod.data());
        execG(*dbl[j]); dqOut(*dbl[j], 0, img.data()); dqOut(*dbl[j], 1, txt.data());
      }
      memcpy(xm.data(), txt.data(), TXT*D*sizeof(float));            // cat([txt, img])
      memcpy(xm.data() + TXT*D, img.data(), IMG*D*sizeof(float));
      for (int gi = 0; gi < 3; gi++) {                              // singles 0..19 in 3 contexts
        if (isRes[gi]) runSingles(resCtx[gi], groups[gi].first, groups[gi].last);
        else { Ctx c = loadCtx(be, dev, BP(groups[gi].path).c_str()); runSingles(c, groups[gi].first, groups[gi].last); freeCtx(c); }
      }
      memcpy(imgtok.data(), xm.data() + TXT*D, IMG*D*sizeof(float)); // img = xm[512:]
      qIn(*epi, 0, imgtok.data()); qIn(*epi, 1, sv.data());
      execG(*epi); dqOut(*epi, 0, vel.data());
      double dsig = SIG[step+1] - SIG[step];
      for (size_t i = 0; i < x.size(); i++) x[i] += (float)dsig * vel[i];   // Euler
      fprintf(stderr, "[step %d] %.0f ms\n", step, now_ms()-ts);
    }
    double tc1 = now_ms();
    for (int gi = 0; gi < 3; gi++) if (isRes[gi]) freeCtx(resCtx[gi]);
    freeCtx(Rc);
    fprintf(stderr, "[chain] %d step(s) in %.0f ms => %.0f ms/step\n", steps, tc1 - tc0, (tc1 - tc0) / steps);
    const int HW = 32;                                              // unpatchify [1024,128]->[128,32,32]
    std::vector<float> up(128 * HW * HW);
    for (int t = 0; t < IMG; t++) { int i = t / HW, j = t % HW; for (int c = 0; c < 128; c++) up[c*HW*HW + i*HW + j] = x[t*128 + c]; }
    FILE* f = fopen(IP("final_chain.raw").c_str(), "wb"); fwrite(up.data(), 4, up.size(), f); fclose(f);
    printf("DONE %d steps (merged, %d resident sgl), %.0f ms total (%.0f ms/step), wrote final_chain.raw (%zu floats)\n", steps, residentSgl, tc1-tc0, (tc1-tc0)/steps, up.size());
    return 0;
  }

  // FULLY STREAMED residency (argv[4]=resident budget, default 0=stream all). Each block is
  // loaded->run->freed so peak RAM is ONE context (~500 MB), not 6.9 GB. Loading all 27
  // resident OOM-reboots the phone; even 1.6 GB resident is risky right after a reboot.
  // resident>=1 keeps pro+doubles+epi resident (~1.6 GB) once we know there's headroom.
  std::string proPath = BP("proq1.bin"), epiPath = BP("epiq1.bin");
  std::vector<std::string> dblPath, sglPath;
  for (int j=0;j<5;j++)  dblPath.push_back(BP("dbl"+std::to_string(j)+"q1.bin"));
  for (int j=0;j<20;j++) sglPath.push_back(BP("sgl"+std::to_string(j)+"q1.bin"));
  // optional resident set
  Graph proR, epiR; std::vector<Graph> dblR;
  if (resident) {
    double tl0 = now_ms();
    proR = loadGraph(be, dev, proPath.c_str());
    for (int j=0;j<5;j++) dblR.push_back(loadGraph(be, dev, dblPath[j].c_str()));
    epiR = loadGraph(be, dev, epiPath.c_str());
    fprintf(stderr, "[load] pro+5dbl+epi resident in %.0f ms\n", now_ms() - tl0);
  } else fprintf(stderr, "[load] streaming ALL blocks (peak ~1 context)\n");

  const int D = 3072, IMG = 1024, TXT = 512, S = 1536;
  auto x = rdf(IP("latent.raw").c_str());          // patched [1024,128]
  auto context = rdf(IP("context.raw").c_str());   // [512,7680]
  auto cos = rdf(IP("cos.raw").c_str()), sin = rdf(IP("sin.raw").c_str());  // [1536,64]
  std::vector<float> img(IMG*D), txt(TXT*D), imgmod(6*D), txtmod(6*D), smod(3*D), sv(D),
                     xm(S*D), imgtok(IMG*D), vel(IMG*128), temb(256);
  std::vector<double> SIG; flux_sigmas(steps, 1024.0, SIG);   // schedule MUST match step count (3-step != first 3 of 4-step)
  fprintf(stderr, "[sched] sigmas %.4f %.4f %.4f %.4f %.4f\n", SIG[0],SIG[1],SIG[2],SIG[3],SIG[4]);

  // load->run->free if not resident; reuse resident graph otherwise
  auto withBlock = [&](Graph* res, const std::string& path, auto&& fn) {
    if (res && res->ctx) { fn(*res); }
    else { Graph g = loadGraph(be, dev, path.c_str()); fn(g); freeGraph(g); }
  };
  Graph* proRp = resident ? &proR : nullptr;
  Graph* epiRp = resident ? &epiR : nullptr;

  double tc0 = now_ms();
  for (int step = 0; step < steps; step++) {
    double ts = now_ms();
    temb_of(SIG[step], temb.data());
    withBlock(proRp, proPath, [&](Graph& P){
      qIn(P, 0, x.data()); qIn(P, 1, context.data()); qIn(P, 2, temb.data());
      execG(P);
      dqOut(P, 0, img.data()); dqOut(P, 1, txt.data()); dqOut(P, 2, imgmod.data());
      dqOut(P, 3, txtmod.data()); dqOut(P, 4, smod.data()); dqOut(P, 5, sv.data());
    });
    for (int j = 0; j < 5; j++) withBlock(resident ? &dblR[j] : nullptr, dblPath[j], [&](Graph& G){
      qIn(G, 0, img.data()); qIn(G, 1, txt.data()); qIn(G, 2, cos.data()); qIn(G, 3, sin.data()); qIn(G, 4, imgmod.data()); qIn(G, 5, txtmod.data());
      execG(G); dqOut(G, 0, img.data()); dqOut(G, 1, txt.data());
    });
    memcpy(xm.data(), txt.data(), TXT*D*sizeof(float));            // cat([txt, img])
    memcpy(xm.data() + TXT*D, img.data(), IMG*D*sizeof(float));
    for (int j = 0; j < 20; j++) withBlock(nullptr, sglPath[j], [&](Graph& G){   // singles always streamed
      qIn(G, 0, xm.data()); qIn(G, 1, cos.data()); qIn(G, 2, sin.data()); qIn(G, 3, smod.data());
      execG(G); dqOut(G, 0, xm.data());
    });
    memcpy(imgtok.data(), xm.data() + TXT*D, IMG*D*sizeof(float)); // img = xm[512:]
    withBlock(epiRp, epiPath, [&](Graph& E){
      qIn(E, 0, imgtok.data()); qIn(E, 1, sv.data());
      execG(E); dqOut(E, 0, vel.data());
    });
    double dsig = SIG[step+1] - SIG[step];
    for (size_t i = 0; i < x.size(); i++) x[i] += (float)dsig * vel[i];   // Euler
    fprintf(stderr, "[step %d] %.0f ms\n", step, now_ms()-ts);
  }
  double tc1 = now_ms();
  fprintf(stderr, "[chain] %d step(s) in %.0f ms => %.0f ms/step\n", steps, tc1 - tc0, (tc1 - tc0) / steps);

  // unpatchify [1024,128] -> [128,32,32] so it's decode-ready for sd-cli SD_LOAD_LATENT
  const int HW = 32;
  std::vector<float> up(128 * HW * HW);
  for (int t = 0; t < IMG; t++) { int i = t / HW, j = t % HW; for (int c = 0; c < 128; c++) up[c*HW*HW + i*HW + j] = x[t*128 + c]; }
  FILE* f = fopen(IP("final_chain.raw").c_str(), "wb"); fwrite(up.data(), 4, up.size(), f); fclose(f);
  printf("DONE %d steps, %.0f ms total (%.0f ms/step), wrote final_chain.raw (%zu floats, unpatchified)\n", steps, tc1-tc0, (tc1-tc0)/steps, up.size());
  return 0;
}

int main(int argc, char** argv) {
  const char* binsDir = argc > 1 ? argv[1] : ".";
  const char* ioDir   = argc > 2 ? argv[2] : ".";
  int steps = argc > 3 ? atoi(argv[3]) : 4;
  // 0 = stream 27 separate bins; 1 = pro+5dbl+epi resident + stream singles;
  // 2 = MERGED 3-context mode (Rq1 + sglAq1 + sglBq1), ~7 context-creates for in-app use.
  int resident = argc > 4 ? atoi(argv[4]) : 0;
  return run_chain(binsDir, ioDir, steps, resident);
}

#ifdef BONSAI_JNI
#include <jni.h>
#include <android/log.h>
#include <unistd.h>
#include <pthread.h>

static void* logpipe_reader(void* arg) {
  int fd = (int)(intptr_t)arg; char buf[1024]; std::string line; ssize_t n;
  while ((n = read(fd, buf, sizeof(buf) - 1)) > 0)
    for (ssize_t i = 0; i < n; i++) {
      if (buf[i] == '\n') { __android_log_print(ANDROID_LOG_INFO, "bonsaichain", "%s", line.c_str()); line.clear(); }
      else line += buf[i];
    }
  return nullptr;
}
// Pipe stdout+stderr (unbuffered) to logcat so a stall shows the last line live.
static void redirect_stdio_to_logcat() {
  static bool done = false; if (done) return; done = true;
  setvbuf(stdout, nullptr, _IONBF, 0); setvbuf(stderr, nullptr, _IONBF, 0);
  int pfd[2]; if (pipe(pfd) != 0) return;
  dup2(pfd[1], 1); dup2(pfd[1], 2); close(pfd[1]);
  pthread_t t; pthread_create(&t, nullptr, logpipe_reader, (void*)(intptr_t)pfd[0]); pthread_detach(t);
}

extern "C" JNIEXPORT jint JNICALL
Java_now_shorty_bonsai_MainActivity_nativeRunChain(JNIEnv* env, jobject, jstring jbins, jstring jnio, jint steps, jint resident) {
  redirect_stdio_to_logcat();
  const char* b = env->GetStringUTFChars(jbins, nullptr);
  const char* n = env->GetStringUTFChars(jnio, nullptr);
  int rc = run_chain(b, n, (int)steps, (int)resident);
  env->ReleaseStringUTFChars(jbins, b);
  env->ReleaseStringUTFChars(jnio, n);
  return rc;
}

// De-risk: load TWO contexts (pro then sgl0) in-process — the exact point QNN 2.46
// stalled (right after the 1st context). If both load + exec, the runtime handles
// multi-context in-app. binsdir holds 2.41-built proq1.bin + sgl0q1.bin.
extern "C" JNIEXPORT jstring JNICALL
Java_now_shorty_bonsai_MainActivity_nativeCtxTest(JNIEnv* env, jobject, jstring jdir, jstring jbins) {
  redirect_stdio_to_logcat();
  const char* libdir = env->GetStringUTFChars(jdir, nullptr);
  const char* binsdir = env->GetStringUTFChars(jbins, nullptr);
  std::string ld = libdir, bd = binsdir;
  env->ReleaseStringUTFChars(jdir, libdir); env->ReleaseStringUTFChars(jbins, binsdir);
  void* lib = dlopen((ld + "/libQnnHtp.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!lib) return env->NewStringUTF("dlopen libQnnHtp FAIL");
  const QnnInterface_t** prov = nullptr; uint32_t np = 0;
  ((Qnn_ErrorHandle_t(*)(const QnnInterface_t***, uint32_t*))dlsym(lib, "QnnInterface_getProviders"))(&prov, &np);
  q = prov[0]->QNN_INTERFACE_VER_NAME;
  void* sysl = dlopen((ld + "/libQnnSystem.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
  const QnnSystemInterface_t** sp = nullptr; uint32_t nsp = 0;
  ((Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t***, uint32_t*))dlsym(sysl, "QnnSystemInterface_getProviders"))(&sp, &nsp);
  s = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  Qnn_LogHandle_t log = nullptr; q.logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &log);
  Qnn_BackendHandle_t be = nullptr; if (q.backendCreate(log, nullptr, &be)) return env->NewStringUTF("backendCreate FAIL");
  Qnn_DeviceHandle_t dev = nullptr; if (q.deviceCreate(log, nullptr, &dev)) return env->NewStringUTF("deviceCreate FAIL");
  fprintf(stderr, "[ctxtest] device ok, loading pro (ctx 1)...\n");
  Graph g1 = loadGraph(be, dev, (bd + "/proq1.bin").c_str());
  fprintf(stderr, "[ctxtest] pro ok, loading sgl0 (ctx 2 = the 2.46 stall point)...\n");
  Graph g2 = loadGraph(be, dev, (bd + "/sgl0q1.bin").c_str());
  fprintf(stderr, "[ctxtest] BOTH contexts loaded, exec sgl0...\n");
  execG(g2);
  fprintf(stderr, "[ctxtest] exec ok\n");
  return env->NewStringUTF("2.41 OK: 2 contexts loaded + exec in-process (past the 2.46 stall)");
}
#endif
