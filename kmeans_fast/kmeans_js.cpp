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
double *centroids;    // K x MAX_BIGRAM
double *dists;        // num_recs x K

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
    double *centroid_entropy = new double[cend-cstart]();
    
    for (size_t i = 0; i < cend-cstart; i++) {
        for (int j = 0; j < MAX_BIGRAM; j++) {
            double x = cent[index(i,j,MAX_BIGRAM)];
            if (x != 0) centroid_entropy[i] += x*log(x);
        }
        centroid_entropy[i] = -centroid_entropy[i];
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
            rhs = (obs_entropy+centroid_entropy[c]) / 2.0;

            // lhs = H((A+B)/2)
            // Densify the observation
            densify(scratch, *sphist_indices[i], *sphist[i]);
            
            // Add in the centroid
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
        //if ((i+1) % 10000 == 0) fprintf(stderr, "%d / %d distances computed.\n", i+1, end-start);
    }

    delete[] centroid_entropy;
}

void *thread_func(void *arg) {
    thread_args_t a = *(thread_args_t *)arg;
    batch_js_dist(a.start, a.end, a.cent, a.cstart, a.cend, a.out);
    //batch_euclid_dist(a.start, a.end);
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

// Checks whether the ith centroid is distinct from the previous i-1.
bool centroid_distinct(int i) {
    for (int j = 0; j < i; j++) {
        bool equal = true;
        for (int ri = 0; ri < MAX_BIGRAM; ri++) {
            if (centroids[index(i,ri,MAX_BIGRAM)] != centroids[index(j,ri,MAX_BIGRAM)]) {
                equal = false;
                break;
            }
        }
        if (equal) return false;
    }
    return true;
}

// Initializes the centroids. For now using random init, later using kmeans++
void init_centroids() {
    srand(time(NULL));
    // Random for now. Do kmeans++ later.
    int *which = new int[num_means];

    for(int i = 0; i < num_means; i++) {
        do {
            int r = rand() % num_recs;
            which[i] = r;
            densify(centroids+(i*MAX_BIGRAM), *sphist_indices[r], *sphist[r]);
        } while (!centroid_distinct(i));
    }
    fprintf(stderr, "Means chosen:");
    for(int i = 0; i < num_means; i++) fprintf(stderr, " %d", which[i]);
    fprintf(stderr, "\n");
}

void init_centroids_kmpp() {
    time_t start = time(NULL);

    int n_local_trials = 2 + log(num_means);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);
    
    // First chosen uniform randomly
    fprintf(stderr, "Choosing centroid %d...\n", 0);
    int cand = dis(gen) * num_recs;
    densify(centroids, *sphist_indices[cand], *sphist[cand]);

    // Compute (squared) distances to initial centroid
    double *closest_dist_sq = new double[num_recs]();
    parallel_dists(centroids, 0, 1, closest_dist_sq);

    double current_pot = 0.0;
    for (size_t i = 0; i < num_recs; i++) {
        double x = closest_dist_sq[i];
        closest_dist_sq[i] = x*x;
    }
    for (size_t i = 0; i < num_recs; i++) {
        current_pot += closest_dist_sq[i];
    }

    // Compute the rest
    for (int c = 1; c < num_means; c++) {
        fprintf(stderr, "Choosing centroid %d...\n", c);
        double *local_centroids = ARRAY_ALLOC(n_local_trials, MAX_BIGRAM, double);
        CHECKFAIL(local_centroids, "calloc");

        double *distance_to_candidates = ARRAY_ALLOC(num_recs, n_local_trials, double);
        CHECKFAIL(distance_to_candidates, "calloc");

        double *rand_vals = new double[n_local_trials]();
        int *candidate_ids = new int[n_local_trials]();
        double *cumsum = new double[num_recs]();

        for(int i = 0; i < n_local_trials; i++)
            rand_vals[i] = dis(gen) * current_pot;

        // Cumulative sum over squared distances
        std::partial_sum(closest_dist_sq, closest_dist_sq+num_recs, cumsum, std::plus<double>());
        
        for (int i = 0; i < n_local_trials; i++)
            candidate_ids[i] = std::distance(cumsum, std::lower_bound(cumsum, cumsum+num_recs, rand_vals[i]));

        for (int i = 0; i < n_local_trials; i++)
            densify(local_centroids+(i*MAX_BIGRAM), *sphist_indices[candidate_ids[i]], *sphist[candidate_ids[i]]);

        // Distances to candidates
        parallel_dists(local_centroids, 0, n_local_trials, distance_to_candidates);
        // Squared
        for (size_t i = 0; i < num_recs; i++) {
            for(int j = 0; j < n_local_trials; j++) {
                double x = distance_to_candidates[index(i,j,n_local_trials)];
                distance_to_candidates[index(i,j,n_local_trials)] = x*x;
            }
        }

        // Determine which candidate is the best
        int best_candidate = -1;
        double best_pot = -1;
        double *best_dist_sq = new double[num_recs]();
        for (int trial = 0; trial < n_local_trials; trial++) {
            double *new_dist_sq = new double[num_recs]();
            double new_pot = 0.0;
            for (size_t i = 0; i < num_recs; i++) {
                new_dist_sq[i] = std::min(closest_dist_sq[i],
                        distance_to_candidates[index(i,trial,n_local_trials)]);
            }
            for (size_t i = 0; i < num_recs; i++) {
                new_pot += new_dist_sq[i];
            }

            if ((best_candidate == -1) || (new_pot < best_pot)) {
                best_candidate = candidate_ids[trial];
                best_pot = new_pot;
                memcpy(best_dist_sq, new_dist_sq, sizeof(double)*num_recs);
            }
            
            delete[] new_dist_sq;
        }
        
        // Permanently add the best candidate
        densify(centroids+(c*MAX_BIGRAM), *sphist_indices[best_candidate], *sphist[best_candidate]);
        current_pot = best_pot;
        memcpy(closest_dist_sq, best_dist_sq, sizeof(double)*num_recs);

        // Cleanup
        delete[] best_dist_sq;
        delete[] rand_vals;
        delete[] candidate_ids;
        delete[] cumsum;
        free(local_centroids);
        free(distance_to_candidates);
    }

    delete[] closest_dist_sq;
    fprintf(stderr, "Chose %d means in %ld seconds.\n", num_means, time(NULL)-start);
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

void tfidf_adjust() {
    // Compute IDF(t,D) for each bigram
    double idf[MAX_BIGRAM] = {0.0};
    int termcount[MAX_BIGRAM] = {0};
    for (size_t i = 0; i < num_recs; i++) {
        for(std::vector<uint16_t>::iterator ind = sphist_indices[i]->begin();
                ind != sphist_indices[i]->end(); ++ind) {
            termcount[*ind] += 1;
        }
    }
    for (int t = 0; t < MAX_BIGRAM; t++) {
        if(termcount[t] != 0)
            idf[t] = log(num_recs / (double)termcount[t]);
        else
            fprintf(stderr, "Term %#x not in corpus\n", t);
    }

    fprintf(stderr, "Term: [ ");
    for (int t = 0; t < MAX_BIGRAM; t++)
        fprintf(stderr, "%d ", termcount[t]);
    fprintf(stderr, "]\n");

    fprintf(stderr, "IDF: [ ");
    for (int t = 0; t < MAX_BIGRAM; t++)
        fprintf(stderr, "%f ", idf[t]);
    fprintf(stderr, "]\n");

    // Multiply by the term frequencies
    for (size_t i = 0; i < num_recs; i++) {
        std::vector<uint16_t>::iterator ind = sphist_indices[i]->begin();
        std::vector<double>::iterator val = sphist[i]->begin();
        for( ; // Init done outside loop
             (ind != sphist_indices[i]->end()) && (val != sphist[i]->end());
             ++ind, ++val ) {
            *val *= idf[*ind];
        }
    }
}

void print_centroids() {
    for (int i = 0; i < num_means; i++) {
        fprintf(stderr, "%d: [", i);
        for (int j = 0; j < MAX_BIGRAM; j++) {
            fprintf(stderr, " %e,", centroids[index(i,j,MAX_BIGRAM)]);
            if (j == 5) {
                fprintf(stderr, " ...,");
                j += 65524;
            }
        }
        fprintf(stderr, " ]\n"); 
    }
}

void print_dists() {
    for (size_t i = 0; i < num_recs; i++) {
        fprintf(stderr, "%ld: [", i);
        for (int j = 0; j < num_means; j++) {
            fprintf(stderr, " %e,", dists[index(i,j,num_means)]);
        }
        fprintf(stderr, " ]\n"); 
    }
}

double centroid_sparsity() {
    double total = 0, nonzero = 0;
    for (int i = 0; i < num_means; i++) {
        for (int j = 0; j < MAX_BIGRAM; j++) {
            if (centroids[index(i,j,MAX_BIGRAM)] != 0) nonzero += 1;
            total += 1;
        }
    }
    
    return nonzero / total;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bigram_file> <k>\n", argv[0]);
        return 1;
    }

    struct stat st;
    CHECKFAIL(!stat(argv[1],&st), "stat");
    
    FILE *f = fopen(argv[1], "rb");
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
    
    // Init the RNG from /dev/urandom
    FILE *rng = fopen("/dev/urandom", "rb");
    CHECKFAIL(rng, "fopen");
    long int seed;
    nread = fread(&seed, sizeof(seed), 1, rng);
    CHECKFAIL(nread, "fread");
    fclose(rng);
    srand48(seed);
   
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
        //if (num_recs == 8192) break;
        //if(num_recs % 10000 == 0) fprintf(stderr, "Processed %d recs.\n", num_recs);
    }

    //fprintf(stderr, "Applying TF-IDF transformation...\n");
    //tfidf_adjust();
    fprintf(stderr, "Normalizing...\n");
    normalize();

    // Allocate centroids
    num_means = atoi(argv[2]);
    centroids = ARRAY_ALLOC(num_means, MAX_BIGRAM, double);
    CHECKFAIL(centroids, "calloc");

    // Allocate distance matrix
    dists = ARRAY_ALLOC(num_recs, num_means, double);
    CHECKFAIL(dists, "calloc");

    // Allocate assignemnts
    int    *assignments = new int[num_recs]();
 
    // Init
    //init_centroids();
    init_centroids_kmpp();
    print_centroids();
    fprintf(stderr, "Centroid sparsity: %f\n", centroid_sparsity());

    int iter = 0;
    while (true) {
        time_t start_time = time(NULL);
        bool changed = false;
        int num_changed = 0;
        fprintf(stderr, "Iteration %d... ", iter++);
        parallel_dists(centroids, 0, num_means, dists);
        //print_dists();

        // Find closest centroid for each point
        for (size_t i = 0; i < num_recs; i++) {
            int new_assignment = 0;
            std::vector<double> v;
            for (int c = 0; c < num_means; c++) {
                v.push_back(dists[index(i,c,num_means)]);
            }

            new_assignment = std::distance(v.begin(), std::min_element(v.begin(), v.end()));

            if (new_assignment != assignments[i]) {
                assignments[i] = new_assignment;
                changed = true;
                num_changed++;
            }
        }
        
        // Check for termination
        if (!changed) {
            fprintf(stderr, "No assignments changed, we are finished!\n");
            break;
        }

        // Update centroids
        int *cluster_sizes = new int[num_means]();
        double *new_centroids = ARRAY_ALLOC(num_means, MAX_BIGRAM, double);
        CHECKFAIL(new_centroids, "calloc");

        // Sum of cluster members
        for(size_t i = 0; i < num_recs; i++) {
            int c = assignments[i];
            std::vector<uint16_t>::iterator ind = sphist_indices[i]->begin();
            std::vector<double>::iterator val = sphist[i]->begin();
            for( ; // Init done outside loop
                 (ind != sphist_indices[i]->end()) && (val != sphist[i]->end());
                 ++ind, ++val ) {
                new_centroids[index(c,(*ind),MAX_BIGRAM)] += *val;
            }
            cluster_sizes[c] += 1;
        }
        
        // Divide through by cluster size
        for (int i = 0; i < num_means; i++) {
            for (int j = 0; j < MAX_BIGRAM; j++) {
                new_centroids[index(i,j,MAX_BIGRAM)] /= cluster_sizes[i];
            }
        }

        // Debug: compute change in centroids
        double *centroid_distances = new double[num_means]();
        for (int i = 0; i < num_means; i++) {
            double sum = 0;
            // Euclidean for debugging
            for (int j = 0; j < MAX_BIGRAM; j++) {
                double diff = new_centroids[index(i,j,MAX_BIGRAM)] - centroids[index(i,j,MAX_BIGRAM)];
                sum += diff*diff;
            }
            centroid_distances[i] = sqrt(sum);
        }

        // Compute RSS
        double rss = 0.0;
        for (size_t i = 0; i < num_recs; i++) {
            int c = assignments[i];
            double x = dists[index(i,c,num_means)];
            rss += x*x;
        }

        memcpy(centroids, new_centroids, sizeof(double) * num_means * MAX_BIGRAM);
        
        fprintf(stderr, "done. Took %ld seconds.\n", time(NULL)-start_time);
        fprintf(stderr, "RSS=%e\n", rss);
        fprintf(stderr, "Centroid sparsity: %f\n", centroid_sparsity());
        fprintf(stderr, "%d assignment(s) changed this iteration.\n", num_changed);
        fprintf(stderr, "Centroid change: [ ");
        for (int i = 0; i < num_means; i++) fprintf(stderr, "%f ", centroid_distances[i]);
        fprintf(stderr, "]\n");
        delete[] cluster_sizes;
        delete[] centroid_distances;
        free(new_centroids);
    }

    // All done, print things out
    for (size_t i = 0; i < num_recs; i++) {
        prog_point &pp = meta_info[i];
        printf(ADDRFMT " " ADDRFMT " " ADDRFMT " %f %d\n", pp.caller, pp.pc, pp.cr3,
            dists[index(i,assignments[i],num_means)],
            assignments[i]);
    }

    return 0;
}
