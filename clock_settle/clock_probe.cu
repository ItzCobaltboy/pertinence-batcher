// clock_probe.cu  --  what does the GPU clock really do around a sysfs clock change? (Jetson AGX Xavier, needs root)
//
// A single GPU thread spins and publishes its SM cycle counter (clock64) into pinned, mapped host memory. A reader thread pinned to
// one CPU core reads that counter in a tight loop and stamps every new value with CLOCK_MONOTONIC (the clock the harness uses).
// The counter advances at the real GPU clock, so counter / time = the clock in use at that moment. The main thread, pinned to another
// core, changes the clock through devfreq (the same writes as common/dvfs.py Clock.set) and stamps the write start, middle and end.
// Plotting the clock against time shows when the GPU really changes speed, whether it pauses, and how long the write blocks.
//
// Build:  nvcc -O2 -arch=sm_72 -o clock_probe clock_probe.cu -lpthread
// Run:    sudo ./clock_probe --from 829 --to 1237 --mode busy --repeats 100 --out run_busy_829_1237
//         --mode busy : the kernel is running when the clock changes.
//         --mode idle : no kernel for --idle-ms before the change, the kernel starts right after the write returns.
// Stop every other GPU program first. The original min_freq / max_freq are restored at exit (also on Ctrl+C).

#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------------------------- GPU side
// volatile on both pointers is not optional. Without it the compiler may keep the counter in a register and store it once at
// the end, or load the stop flag once before the loop, and the kernel would never see the host's stop signal.
__global__ void spin(volatile unsigned long long *published, volatile int *stop_flag)
{
    while (*stop_flag == 0) {
        unsigned long long cycles = (unsigned long long)clock64();   // SM cycle counter: its rate IS the GPU clock
        *published = cycles;                                         // one aligned 64-bit store, so the host never reads half of it
        __threadfence_system();                                      // push the store out to memory the CPU can see
    }
}

// ---------------------------------------------------------------------------------------------------------------- host state
// Mapped (zero-copy) memory, allocated without cudaHostAllocWriteCombined: write-combined memory is very slow for the CPU to READ.
static volatile unsigned long long *g_pub = NULL;     // written by the GPU, read by the reader thread
static volatile int *g_stop = NULL;                   // written by the host, read by the GPU
static unsigned long long *g_pub_dev = NULL;          // the same two words as the GPU sees them
static int *g_stop_dev = NULL;

static volatile int g_recording = 0;                  // main thread -> reader thread
static volatile int g_quit = 0;
static volatile int g_rep = 0;
static volatile int g_overflow = 0;
static volatile sig_atomic_t g_interrupted = 0;

struct Sample { int64_t t_ns; unsigned long long cycles; int rep; };
static Sample *g_buf = NULL;
static size_t g_cap = 0;
static size_t g_n = 0;                                // only the reader thread writes it, the main thread reads it after the join
static int64_t g_sample_ns = 20000;

static inline int64_t now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void sleep_ms(double ms)
{
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000.0);
    ts.tv_nsec = (long)((ms - 1000.0 * (double)ts.tv_sec) * 1e6);
    nanosleep(&ts, NULL);
}

static void pin_to_core(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        fprintf(stderr, "warning: could not pin to core %d\n", core);
}

static void on_signal(int) { g_interrupted = 1; }

// ---------------------------------------------------------------------------------------------------------------- reader thread
static void *reader_main(void *arg)
{
    pin_to_core(*(int *)arg);
    unsigned long long last_c = 0;
    int64_t last_t = 0;
    while (!g_quit) {
        if (!g_recording)
            continue;
        unsigned long long c = *g_pub;                // volatile read of the GPU's latest cycle count
        int64_t t = now_ns();                         // stamped right after the read
        if (c != last_c && t - last_t >= g_sample_ns) {   // keep a sample only if the value is new and at least --sample-us after the last
            if (g_n < g_cap) {
                g_buf[g_n].t_ns = t;
                g_buf[g_n].cycles = c;
                g_buf[g_n].rep = g_rep;
                g_n++;
            } else {
                g_overflow = 1;
            }
            last_c = c;
            last_t = t;
        }
    }
    return NULL;
}

// ---------------------------------------------------------------------------------------------------------------- devfreq
struct Devfreq {
    std::string dir;
    std::vector<long long> avail_hz;
    long long min0, max0;                             // the original values, written back at exit
    bool restored;
};
static Devfreq g_dev;

static std::string read_text(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    char line[4096];
    std::string out;
    while (fgets(line, sizeof(line), f)) out += line;
    fclose(f);
    return out;
}

static int write_text(const std::string &path, long long value)
{
    char text[64];
    snprintf(text, sizeof(text), "%lld\n", value);
    int fd = open(path.c_str(), O_WRONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s for writing (root?)\n", path.c_str()); return -1; }
    ssize_t n = write(fd, text, strlen(text));
    int rc = close(fd);
    if (n < 0 || rc < 0) { fprintf(stderr, "write to %s failed\n", path.c_str()); return -1; }
    return 0;
}

static long long nearest_hz(long long mhz)
{
    long long best = g_dev.avail_hz[0];
    for (size_t i = 0; i < g_dev.avail_hz.size(); i++) {
        long long h = g_dev.avail_hz[i];
        long long d1 = h > mhz * 1000000LL ? h - mhz * 1000000LL : mhz * 1000000LL - h;
        long long d2 = best > mhz * 1000000LL ? best - mhz * 1000000LL : mhz * 1000000LL - best;
        if (d1 < d2) best = h;
    }
    return best;
}

// Setup only (not timed): always valid whatever the current min/max are. lowest -> min, then max = hz, then min = hz.
static void pin_clock_safe(long long hz)
{
    write_text(g_dev.dir + "/min_freq", g_dev.avail_hz[0]);
    write_text(g_dev.dir + "/max_freq", hz);
    write_text(g_dev.dir + "/min_freq", hz);
}

// The timed change, same order as common/dvfs.py Clock.set: raising writes max first, lowering writes min first.
static void set_clock(long long cur_hz, long long new_hz, int64_t *t_mid)
{
    if (new_hz >= cur_hz) {
        write_text(g_dev.dir + "/max_freq", new_hz);
        *t_mid = now_ns();
        write_text(g_dev.dir + "/min_freq", new_hz);
    } else {
        write_text(g_dev.dir + "/min_freq", new_hz);
        *t_mid = now_ns();
        write_text(g_dev.dir + "/max_freq", new_hz);
    }
}

static void restore_clocks()
{
    if (g_dev.restored) return;
    g_dev.restored = true;
    write_text(g_dev.dir + "/min_freq", g_dev.avail_hz[0]);
    write_text(g_dev.dir + "/max_freq", g_dev.max0);
    write_text(g_dev.dir + "/min_freq", g_dev.min0);
    fprintf(stderr, "clocks restored: min %lld max %lld\n", g_dev.min0, g_dev.max0);
}

// ---------------------------------------------------------------------------------------------------------------- kernel control
static cudaStream_t g_stream;

static void start_kernel()
{
    *g_stop = 0;
    spin<<<1, 1, 0, g_stream>>>(g_pub_dev, g_stop_dev);
    CK(cudaGetLastError());
}

static void stop_kernel()
{
    *g_stop = 1;
    CK(cudaStreamSynchronize(g_stream));
}

// ---------------------------------------------------------------------------------------------------------------- main
struct Event {
    int rep;
    long long from_hz, to_hz;
    int64_t t_write_start, t_write_mid, t_write_end, t_launch_start, t_launch_end;
};

int main(int argc, char **argv)
{
    std::string devfreq = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b";   // DEVFREQ in common/dvfs.py
    std::string out = "probe";
    std::string mode = "busy";
    long long from_mhz = 829, to_mhz = 1237;
    int repeats = 100, reader_core = 2, writer_core = 3;
    double pre_ms = 30.0, post_ms = 250.0, idle_ms = 200.0;
    double sample_us = 20.0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        bool has = i + 1 < argc;
        if (a == "--devfreq" && has) devfreq = argv[++i];
        else if (a == "--from" && has) from_mhz = atoll(argv[++i]);
        else if (a == "--to" && has) to_mhz = atoll(argv[++i]);
        else if (a == "--mode" && has) mode = argv[++i];
        else if (a == "--repeats" && has) repeats = atoi(argv[++i]);
        else if (a == "--pre-ms" && has) pre_ms = atof(argv[++i]);
        else if (a == "--post-ms" && has) post_ms = atof(argv[++i]);
        else if (a == "--idle-ms" && has) idle_ms = atof(argv[++i]);
        else if (a == "--sample-us" && has) sample_us = atof(argv[++i]);
        else if (a == "--reader-core" && has) reader_core = atoi(argv[++i]);
        else if (a == "--writer-core" && has) writer_core = atoi(argv[++i]);
        else if (a == "--out" && has) out = argv[++i];
        else { fprintf(stderr, "unknown or incomplete argument: %s\n", a.c_str()); return 1; }
    }
    bool busy = (mode == "busy");
    if (!busy && mode != "idle") { fprintf(stderr, "--mode must be busy or idle\n"); return 1; }
    g_sample_ns = (int64_t)(sample_us * 1000.0);

    // devfreq
    g_dev.dir = devfreq;
    g_dev.restored = false;
    std::string avail = read_text(devfreq + "/available_frequencies");
    const char *p = avail.c_str();
    char *end = NULL;
    for (;;) {
        long long v = strtoll(p, &end, 10);
        if (end == p) break;
        g_dev.avail_hz.push_back(v);
        p = end;
    }
    if (g_dev.avail_hz.empty()) { fprintf(stderr, "no available_frequencies\n"); return 1; }
    g_dev.min0 = atoll(read_text(devfreq + "/min_freq").c_str());
    g_dev.max0 = atoll(read_text(devfreq + "/max_freq").c_str());
    atexit(restore_clocks);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    long long hz_from = nearest_hz(from_mhz), hz_to = nearest_hz(to_mhz);
    printf("original min %lld max %lld Hz; probing %lld Hz <-> %lld Hz, mode %s, %d repeats\n", g_dev.min0, g_dev.max0, hz_from, hz_to, mode.c_str(), repeats);

    // CUDA and the two mapped words (kept 64 bytes apart so they do not share a cache line)
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CK(cudaStreamCreate(&g_stream));
    void *mem = NULL;
    CK(cudaHostAlloc(&mem, 128, cudaHostAllocMapped));
    memset(mem, 0, 128);
    g_pub = (volatile unsigned long long *)mem;
    g_stop = (volatile int *)((char *)mem + 64);
    CK(cudaHostGetDevicePointer((void **)&g_pub_dev, (void *)g_pub, 0));
    CK(cudaHostGetDevicePointer((void **)&g_stop_dev, (void *)g_stop, 0));

    // sample buffer: at most one sample per --sample-us while recording
    double window_ms = pre_ms + post_ms + 50.0;
    g_cap = (size_t)((double)repeats * window_ms * 1000.0 / sample_us * 1.3) + 1000;
    g_buf = (Sample *)malloc(g_cap * sizeof(Sample));
    if (!g_buf) { fprintf(stderr, "cannot allocate %zu samples\n", g_cap); return 1; }

    pthread_t reader;
    pthread_create(&reader, NULL, reader_main, &reader_core);
    pin_to_core(writer_core);

    std::vector<Event> events;
    for (int r = 0; r < repeats && !g_interrupted; r++) {
        long long a = (r % 2 == 0) ? hz_from : hz_to;          // even repeats go from -> to, odd repeats go back
        long long b = (r % 2 == 0) ? hz_to : hz_from;
        pin_clock_safe(a);                                      // starting clock, GPU idle
        sleep_ms(300.0);
        g_rep = r;
        if (busy) {
            start_kernel();
            sleep_ms(100.0);                                    // kernel running at the starting clock
        } else {
            sleep_ms(idle_ms);                                  // GPU idle for this long before the change
        }
        g_recording = 1;
        sleep_ms(pre_ms);
        Event ev;
        ev.rep = r;
        ev.from_hz = a;
        ev.to_hz = b;
        ev.t_launch_start = ev.t_launch_end = 0;
        ev.t_write_start = now_ns();
        set_clock(a, b, &ev.t_write_mid);
        ev.t_write_end = now_ns();
        if (!busy) {
            ev.t_launch_start = now_ns();
            start_kernel();
            ev.t_launch_end = now_ns();
        }
        sleep_ms(post_ms);
        g_recording = 0;
        stop_kernel();
        events.push_back(ev);
        if ((r + 1) % 10 == 0) printf("  %d / %d repeats\n", r + 1, repeats);
    }

    g_quit = 1;
    pthread_join(reader, NULL);
    restore_clocks();
    if (g_overflow) fprintf(stderr, "WARNING: the sample buffer overflowed, later samples were lost (use a larger --sample-us)\n");

    std::string f1 = out + "_samples.csv", f2 = out + "_events.csv";
    FILE *fs = fopen(f1.c_str(), "w");
    FILE *fe = fopen(f2.c_str(), "w");
    if (!fs || !fe) { fprintf(stderr, "cannot write the output files\n"); return 1; }
    fprintf(fs, "rep,t_ns,cycles\n");
    for (size_t i = 0; i < g_n; i++)
        fprintf(fs, "%d,%lld,%llu\n", g_buf[i].rep, (long long)g_buf[i].t_ns, g_buf[i].cycles);
    fprintf(fe, "rep,mode,from_hz,to_hz,t_write_start_ns,t_write_mid_ns,t_write_end_ns,t_launch_start_ns,t_launch_end_ns\n");
    for (size_t i = 0; i < events.size(); i++)
        fprintf(fe, "%d,%s,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n", events[i].rep, mode.c_str(), events[i].from_hz, events[i].to_hz,
                (long long)events[i].t_write_start, (long long)events[i].t_write_mid, (long long)events[i].t_write_end,
                (long long)events[i].t_launch_start, (long long)events[i].t_launch_end);
    fclose(fs);
    fclose(fe);
    printf("wrote %s (%zu samples) and %s (%zu events)\n", f1.c_str(), g_n, f2.c_str(), events.size());
    return 0;
}
