/*
 * @author Prasanth Thomas Shaji
 *
 * Clone of the numpy.random.standard_normal
 * implementation
 *
 */

#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>
#include <math.h>

#include "ziggurat.h"

typedef struct {
	uint64_t high;
	uint64_t low;
} pcg128_t;

typedef struct {
	pcg128_t state;
	pcg128_t inc;
} pcg_state_setseq_128;

typedef pcg_state_setseq_128 pcg64_random_t;

typedef struct s_pcg64_state {
	pcg64_random_t *pcg_state;
	int has_uint32;
	uint32_t uinteger;
} pcg64_state;

typedef struct bitgen
{
	pcg64_state *state;
	uint32_t (*next_uint32)(pcg64_state *state);
} bitgen_t;

pcg64_random_t random_state;
pcg64_state state;
bitgen_t bitgen;

static inline pcg128_t PCG_128BIT_CONSTANT(uint64_t high, uint64_t low) {
	pcg128_t result;
	result.high = high;
	result.low = low;
	return result;
}

#define PCG_DEFAULT_MULTIPLIER_HIGH 2549297995355413924ULL
#define PCG_DEFAULT_MULTIPLIER_LOW 4865540595714422341ULL

#define PCG_DEFAULT_MULTIPLIER_128                                             \
	PCG_128BIT_CONSTANT(PCG_DEFAULT_MULTIPLIER_HIGH, PCG_DEFAULT_MULTIPLIER_LOW)
#define PCG_DEFAULT_INCREMENT_128                                              \
	PCG_128BIT_CONSTANT(6364136223846793005ULL, 1442695040888963407ULL)

static inline pcg128_t pcg128_add(pcg128_t a, pcg128_t b) {
	pcg128_t result;

	result.low = a.low + b.low;
	result.high = a.high + b.high + (result.low < b.low);
	return result;
}

static inline void _pcg_mult64(uint64_t x, uint64_t y, uint64_t *z1,
								uint64_t *z0) {
	uint64_t x0, x1, y0, y1;
	uint64_t w0, w1, w2, t;
	/* Lower 64 bits are straightforward clock-arithmetic. */
	*z0 = x * y;

	x0 = x & 0xFFFFFFFFULL;
	x1 = x >> 32;
	y0 = y & 0xFFFFFFFFULL;
	y1 = y >> 32;
	w0 = x0 * y0;
	t = x1 * y0 + (w0 >> 32);
	w1 = t & 0xFFFFFFFFULL;
	w2 = t >> 32;
	w1 += x0 * y1;
	*z1 = x1 * y1 + w2 + (w1 >> 32);
}

static inline pcg128_t pcg128_mult(pcg128_t a, pcg128_t b) {
	uint64_t h1;
	pcg128_t result;

	h1 = a.high * b.low + a.low * b.high;
	_pcg_mult64(a.low, b.low, &(result.high), &(result.low));
	result.high += h1;
	return result;
}

static inline uint64_t pcg_rotr_64(uint64_t value, unsigned int rot) {
	return (value >> rot) | (value << ((-rot) & 63));
}

static inline void pcg_setseq_128_step_r(pcg_state_setseq_128 *rng) {
	rng->state = pcg128_add(pcg128_mult(rng->state, PCG_DEFAULT_MULTIPLIER_128),
							rng->inc);
}

static inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state) {
	return pcg_rotr_64(state.high ^ state.low, state.high >> 58u);
}

static inline uint64_t
pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128 *rng) {
	pcg_setseq_128_step_r(rng);
	return pcg_output_xsl_rr_128_64(rng->state);
}

#define pcg64_random_r pcg_setseq_128_xsl_rr_64_random_r

static inline uint32_t pcg64_next32(pcg64_state *state) {
	uint64_t next;
	if (state->has_uint32) {
		state->has_uint32 = 0;
		return state->uinteger;
	}
	next = pcg64_random_r(state->pcg_state);
	state->has_uint32 = 1;
	state->uinteger = (uint32_t)(next >> 32);
	return (uint32_t)(next & 0xffffffff);
}

bitgen_t init_prng()
{
	// pcg64_random_t
	random_state.state = PCG_128BIT_CONSTANT(0x979c9a98d8462005ULL, 0x7d3e9cb6cfe0549bULL);
	random_state.inc = PCG_128BIT_CONSTANT(0x0000000000000001ULL, 0xda3e39cb94b95bdbULL);

	// random_state.state = PCG_128BIT_CONSTANT(0x1aa1b5345996452dULL, 0x09585eb7a69561e3ULL);
	// random_state.inc = PCG_128BIT_CONSTANT(0x418ddadb3af71a82ULL, 0x588133bc447873a9ULL);

	// pcg64_state
	state.pcg_state = &random_state;
	state.has_uint32 = 0;
	state.uinteger = 0;

	// bitgen_t
	bitgen.state = &state;
	bitgen.next_uint32 = pcg64_next32;

	return bitgen;
}

static inline uint32_t next_uint32(bitgen_t *bitgen_state) {
	return bitgen_state->next_uint32(bitgen_state->state);
}

static inline float next_float(bitgen_t *bitgen_state) {
	return (next_uint32(bitgen_state) >> 8) * (1.0f / 16777216.0f);
}

float random_standard_normal_f(bitgen_t *bitgen_state)
{
	uint32_t r;
	int sign;
	uint32_t rabs;
	int idx;
	float x, xx, yy;
	for (;;)
	{
		/* r = n23sb8 */
		r = next_uint32(bitgen_state);
		idx = r & 0xff;
		sign = (r >> 8) & 0x1;
		rabs = (r >> 9) & 0x0007fffff;
		x = rabs * wi_float[idx];
		if (sign & 0x1)
			x = -x;
		if (rabs < ki_float[idx])
			return x; /* # 99.3% of the time return here */
		if (idx == 0)
		{
			for (;;)
			{
				/* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
				xx = -ziggurat_nor_inv_r_f * log1pf(-next_float(bitgen_state));
				yy = -log1pf(-next_float(bitgen_state));
				if (yy + yy > xx * xx)
					return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r_f + xx)
											   : ziggurat_nor_r_f + xx;
			}
		}
		else
		{
			if (((fi_float[idx - 1] - fi_float[idx]) * next_float(bitgen_state) +
				 fi_float[idx]) < exp(-0.5 * x * x))
				return x;
		}
	}
}

#endif
