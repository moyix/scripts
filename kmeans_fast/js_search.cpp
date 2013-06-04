#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <vector>
#include <algorithm>
#include <random>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>

#define MAX_BIGRAM 65536
#define INIT_SIZE 65536
#define CHECKFAIL(x,y)              do { if (!x) { perror(y); exit(1); } } while (0);
#define ARRAY_ALLOC(rows,cols,typ)  (typ *) calloc(rows*cols,sizeof(typ))

struct thread_args_t {
    size_t start;
    size_t end;
    double *cent;
    size_t cstart;
    size_t cend;
    double *out;
};

struct prog_point32 {
    uint32_t caller;
    uint32_t pc;
    uint32_t cr3;
};

struct prog_point64 {
    uint64_t caller;
    uint64_t pc;
    uint64_t cr3;
};

#if TARGET_SIZE == 64
typedef prog_point64 prog_point;
#define ADDRFMT "%016lx"
#else
typedef prog_point32 prog_point;
#define ADDRFMT "%08x"
#endif

size_t num_recs;
size_t alloc_recs;
prog_point *meta_info;

// sparse histograms
std::vector<double>   **sphist;
std::vector<uint16_t> **sphist_indices;

int num_means;
double *training;    // 1 x MAX_BIGRAM
double *dists;        // num_recs x 1

// Utilities
inline size_t index(size_t row, size_t col, size_t rowsize) {
    return row*rowsize+col;
}

inline void densify(double *dest, std::vector<uint16_t> &indices, std::vector<double> &vals) {
    std::vector<uint16_t>::iterator ind = indices.begin();
    std::vector<double>::iterator val = vals.begin();
    for( ; // Init done outside loop
         (ind != indices.end()) && (val != vals.end());
         ++ind, ++val ) {
        dest[*ind] = *val;
    }
}

// Distance measures
void batch_euclid_dist(size_t start, size_t end, double *cent, size_t cstart, size_t cend, double *out) {
    for(size_t i = start; i < end; i++) {
        for (size_t c = cstart; c < cend; c++) {
            double sum = 0;

            // Densify
            double scratch[MAX_BIGRAM] = {0.0};
            densify(scratch, *sphist_indices[i], *sphist[i]);

            // Subtract the centroid
            for (int j = 0; j < MAX_BIGRAM; j++) {
                double x = cent[index(c,j,MAX_BIGRAM)];
                scratch[j] -= x;
            }
            // Square
            for (int j = 0; j < MAX_BIGRAM; j++) {
                double x = scratch[j];
                scratch[j] = x*x;
            }
            // Sum
            for (int j = 0; j < MAX_BIGRAM; j++) sum += scratch[j];
            
            //fprintf(stderr, "Dist [%ld,%d] = %e\n", i, c, sqrt(sum));
            out[index(i,c,cend-cstart)] = sqrt(sum);
        }
    }
}

// Computes distance from each centroid to the observations[start:end]
void batch_js_dist(size_t start, size_t end, double *cent, size_t cstart, size_t cend, double *out) {
    double *training_entropy = new double[cend-cstart]();
    
    for (size_t i = 0; i < cend-cstart; i++) {
        for (int j = 0; j < MAX_BIGRAM; j++) {
            double x = cent[index(i,j,MAX_BIGRAM)];
            if (x != 0) training_entropy[i] += x*log(x);
        }
        training_entropy[i] = -training_entropy[i];
    }

    for(size_t i = start; i < end; i++) {
        for (size_t c = cstart; c < cend; c++) {
            double scratch[MAX_BIGRAM] = {0.0};
            double lhs, rhs;

            // rhs = (H(A)+H(B))/2
            double obs_entropy = 0.0;
            for (std::vector<double>::iterator x = sphist[i]->begin(); x != sphist[i]->end(); x++) {
                obs_entropy += (*x)*log(*x);
            }
            obs_entropy = -obs_entropy;
            rhs = (obs_entropy+training_entropy[c]) / 2.0;

            // lhs = H((A+B)/2)
            // Densify the observation
            densify(scratch, *sphist_indices[i], *sphist[i]);
            
            // Add in the training
            for (int j = 0; j < MAX_BIGRAM; j++) {
                double x = cent[index(c,j,MAX_BIGRAM)];
                scratch[j] += x;
            }
            // Divide by 2
            for (int j = 0; j < MAX_BIGRAM; j++) {
                scratch[j] /= 2.0;
            }
            // Compute entropy
            lhs = 0.0;
            for (int j = 0; j < MAX_BIGRAM; j++) {
                double x = scratch[j];
                if (x != 0) lhs += x*log(x);
            }
            lhs = -lhs;
            
            out[index(i,c,cend-cstart)] = lhs - rhs;
        }
    }

    delete[] training_entropy;
}

void *thread_func(void *arg) {
    thread_args_t a = *(thread_args_t *)arg;
    batch_js_dist(a.start, a.end, a.cent, a.cstart, a.cend, a.out);
    //batch_euclid_dist(a.start, a.end, a.cent, a.cstart, a.cend, a.out);
    return NULL;
}

void parallel_dists(double *cent, size_t cstart, size_t cend, double *out) {
    size_t start = 0;
    size_t chunk = ceil(num_recs / (double)NR_CPU);
    pthread_t threads[NR_CPU];
    thread_args_t thread_args[NR_CPU];

    for(int i = 0; i < NR_CPU; ++i) {
        start = chunk*i;
        size_t end = std::min(num_recs, start+chunk);
        thread_args[i].start = start;
        thread_args[i].end = end;
        thread_args[i].cent = cent;
        thread_args[i].cstart = cstart;
        thread_args[i].cend = cend;
        thread_args[i].out = out;
        //fprintf(stderr, "Thread %d from %lld sz %lld\n", i, start, sz);
        pthread_create(&threads[i], NULL, thread_func, &thread_args[i]);
    }
    for(int i = 0; i < NR_CPU; ++i) {
        pthread_join(threads[i], NULL);
    }
}

void normalize() {
    for (size_t i = 0; i < num_recs; i++) {
        double row_sum = 0;
        for (std::vector<double>::iterator it = sphist[i]->begin(); it != sphist[i]->end(); ++it)
            row_sum += *it;
        for (std::vector<double>::iterator it = sphist[i]->begin(); it != sphist[i]->end(); ++it)
            *it /= row_sum;
    }
}


bool paircomparator ( const std::pair<double,size_t>& l, const std::pair<double,size_t>& r) {
    return l.first < r.first;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <training> <bigram_file>\n", argv[0]);
        return 1;
    }

    struct stat st;
    CHECKFAIL(!stat(argv[2],&st), "stat");
    
    FILE *f = fopen(argv[2], "rb");
    CHECKFAIL(f, "fopen");
    
    // 32 or 64 bit bigrams?
    uint32_t word_size = 0;
    int nread = fread(&word_size, sizeof(uint32_t), 1, f);
    if (!nread) {
        perror("fread");
        return 1;
    }

    if (word_size != 8 && word_size != 4) {
        fprintf(stderr, "Invalid word size %u; are you sure this is a bigram file?\n", word_size);
        return 1;
    }
    if (word_size*8 != TARGET_SIZE) {
        fprintf(stderr, "Error: using wrong binary %s (%d-bit executable on %d-bit bigram file)\n",
                argv[0], TARGET_SIZE, word_size*8);
        return 1;
    }
    
    // Start out with space for 2**16 entries; we'll double if necessary
    alloc_recs = INIT_SIZE;
    meta_info = (prog_point *) malloc(alloc_recs * sizeof(prog_point));
    CHECKFAIL(meta_info, "malloc");
    sphist = (std::vector<double> **) malloc(alloc_recs * sizeof(std::vector<double> **));
    CHECKFAIL(sphist, "malloc");
    sphist_indices = (std::vector<uint16_t> **) malloc(alloc_recs * sizeof(std::vector<uint16_t> **));
    CHECKFAIL(sphist_indices, "malloc");
    
    num_recs = 0;
    while (!feof(f)) {
        if (ftell(f) == st.st_size) break;

        if (num_recs >= alloc_recs) {
            alloc_recs *= 2;
            meta_info = (prog_point *) realloc(meta_info, alloc_recs * sizeof(prog_point));
            CHECKFAIL(meta_info, "realloc");
            sphist = (std::vector<double> **) realloc(sphist, alloc_recs * sizeof(std::vector<double> **));
            CHECKFAIL(sphist, "realloc");
            sphist_indices = (std::vector<uint16_t> **) realloc(sphist_indices, alloc_recs * sizeof(std::vector<uint16_t> **));
            CHECKFAIL(sphist_indices, "realloc");
        }
        prog_point pp;
        uint32_t nbins;

        nread = fread(&pp, sizeof(pp), 1, f);
        CHECKFAIL(nread, "fread");
        nread = fread(&nbins, sizeof(nbins), 1, f);
        CHECKFAIL(nread, "fread");
        
        if (!nbins) continue;

        meta_info[num_recs] = pp;
        sphist[num_recs] = new std::vector<double>;
        sphist_indices[num_recs] = new std::vector<uint16_t>();
        for (uint32_t i = 0; i < nbins; i++) {
            uint16_t key;
            uint32_t val;
            nread = fread(&key, sizeof(key), 1, f);
            CHECKFAIL(nread, "fread");
            nread = fread(&val, sizeof(val), 1, f);
            CHECKFAIL(nread, "fread");
            sphist_indices[num_recs]->push_back(key);
            sphist[num_recs]->push_back(val); // note: implicit cast to double
        }
        num_recs++;
    }

    fprintf(stderr, "Normalizing...\n");
    normalize();

    // Allocate training
    training = ARRAY_ALLOC(1, MAX_BIGRAM, double);
    CHECKFAIL(training, "calloc");

    // Allocate distance matrix
    dists = ARRAY_ALLOC(num_recs, 1, double);
    CHECKFAIL(dists, "calloc");

    // Read in training and normalize it
    int64_t itrain[MAX_BIGRAM] = {0};
    FILE *trainf = fopen(argv[1], "rb");
    CHECKFAIL(trainf, "fopen");
    nread = fread(itrain, sizeof(int64_t), MAX_BIGRAM, trainf);
    CHECKFAIL((nread == MAX_BIGRAM), "fread");
    double trainsum = 0;
    for (int i = 0; i < MAX_BIGRAM; i++) training[i] = (double) itrain[i];
    for (int i = 0; i < MAX_BIGRAM; i++) trainsum += training[i];
    for (int i = 0; i < MAX_BIGRAM; i++) training[i] /= trainsum;
    //for (int i = 0; i < MAX_BIGRAM; i++) if (training[i] != 0.0) fprintf(stderr, "%f ", training[i]); fprintf(stderr, "\n");

    // Compute the distances
    parallel_dists(training, 0, 1, dists);

    std::vector< std::pair<double,size_t> > distindices;
    for (size_t i = 0; i < num_recs; i++) distindices.push_back(std::make_pair(dists[i],i));
    std::sort(distindices.begin(), distindices.end(), paircomparator);

    // All done, print things out
    for (auto distpair : distindices) {
        prog_point &pp = meta_info[distpair.second];
        printf(ADDRFMT " " ADDRFMT " " ADDRFMT " %f\n", pp.caller, pp.pc, pp.cr3, distpair.first);
    }

    return 0;
}
