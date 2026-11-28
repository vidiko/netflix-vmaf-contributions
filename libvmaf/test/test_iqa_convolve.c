/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <math.h>
#include <string.h>

#include "test.h"
#include "feature/iqa/convolve.h"

#define W 7
#define H 6

static int almost_equal(float a, float b)
{
    return isfinite(a) && isfinite(b) && fabsf(a - b) < 1e-4f;
}

static void fill_image(float *img)
{
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y * W + x] = (float)((y * 13 + x * 37 + 5) % 17) - 6.0f;
}

/* Direct 2D reference matching _calc_scale + 2D inner product. */
static void convolve_2d_ref(const float *img, int w, int h,
                            const struct _kernel *k,
                            float *dst, int *rw, int *rh)
{
    const int uc = k->w / 2;
    const int vc = k->h / 2;
    const int kw_even = (k->w & 1) ? 0 : 1;
    const int kh_even = (k->h & 1) ? 0 : 1;
    const int dst_w = w - k->w + 1;
    const int dst_h = h - k->h + 1;

    double ksum = 0.0;
    float scale = 1.0f;
    if (!k->normalized) {
        for (int i = 0; i < k->w * k->h; i++)
            ksum += k->kernel[i];
        if (ksum != 0.0)
            scale = (float)(1.0 / ksum);
    }

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            double sum = 0.0;
            int k_offset = 0;
            const int ky = y + vc;
            const int kx = x + uc;
            for (int v = -vc; v <= vc - kh_even; v++) {
                const int img_offset = (ky + v) * w + kx;
                for (int u = -uc; u <= uc - kw_even; u++, k_offset++)
                    sum += img[img_offset + u] * k->kernel[k_offset];
            }
            dst[y * dst_w + x] = (float)(sum * scale);
        }
    }
    *rw = dst_w;
    *rh = dst_h;
}

static void build_2d_kernel(const float *k_h, int kw, const float *k_v, int kh,
                            float *k_2d)
{
    for (int v = 0; v < kh; v++)
        for (int u = 0; u < kw; u++)
            k_2d[v * kw + u] = k_v[v] * k_h[u];
}

/* A non-normalized separable kernel must match both the equivalent normalized
 * separable kernel and the 2D convolution control. */
static char *test_convolve_unnormalized_separable(void)
{
    float k_h[3] = {1.0f, 2.0f, 1.0f};
    float k_v[3] = {1.0f, 4.0f, 1.0f};
    float k_2d[9];
    float n_h[3], n_v[3], n_2d[9];
    float img[W * H], out[W * H], norm_out[W * H], ref_2d[W * H], filt_2d[W * H];
    int rw = -1, rh = -1, ref_rw = -1, ref_rh = -1;

    for (int i = 0; i < 3; i++) {
        n_h[i] = k_h[i] / 4.0f;
        n_v[i] = k_v[i] / 6.0f;
    }
    build_2d_kernel(k_h, 3, k_v, 3, k_2d);
    build_2d_kernel(n_h, 3, n_v, 3, n_2d);

    struct _kernel unnormalized = {
        .kernel = k_2d, .kernel_h = k_h, .kernel_v = k_v,
        .w = 3, .h = 3, .normalized = 0, .bnd_opt = KBND_SYMMETRIC,
    };
    struct _kernel normalized = {
        .kernel = n_2d, .kernel_h = n_h, .kernel_v = n_v,
        .w = 3, .h = 3, .normalized = 1, .bnd_opt = KBND_SYMMETRIC,
    };
    /* Planted negative control: marking non-unit weights normalized=1 skips
     * normalization and scales output by 24x instead of 1x. */
    struct _kernel unscaled_weights = {
        .kernel = k_2d, .kernel_h = k_h, .kernel_v = k_v,
        .w = 3, .h = 3, .normalized = 1, .bnd_opt = KBND_SYMMETRIC,
    };

    fill_image(img);
    convolve_2d_ref(img, W, H, &unnormalized, ref_2d, &ref_rw, &ref_rh);
    mu_assert("unexpected 2d ref width", ref_rw == W - 2);
    mu_assert("unexpected 2d ref height", ref_rh == H - 2);

    _iqa_convolve(img, W, H, &normalized, norm_out, &rw, &rh);
    mu_assert("unexpected output width", rw == W - 2);
    mu_assert("unexpected output height", rh == H - 2);

    memset(out, 0, sizeof(out));
    rw = rh = -1;
    _iqa_convolve(img, W, H, &unnormalized, out, &rw, &rh);
    mu_assert("unexpected unnormalized output width", rw == W - 2);
    mu_assert("unexpected unnormalized output height", rh == H - 2);

    int err = _iqa_img_filter(img, W, H, &unnormalized, filt_2d);
    mu_assert("_iqa_img_filter failed", err == 0);

    for (int y = 0; y < rh; y++) {
        for (int x = 0; x < rw; x++) {
            const int idx = y * rw + x;
            mu_assert("unnormalized separable disagrees with normalized kernel",
                      almost_equal(out[idx], norm_out[idx]));
            mu_assert("unnormalized separable disagrees with 2D control",
                      almost_equal(out[idx], ref_2d[idx]));
            mu_assert("unnormalized separable disagrees with _iqa_img_filter interior",
                      almost_equal(out[idx], filt_2d[(y + 1) * W + (x + 1)]));
        }
    }

    /* Constant non-zero image stays constant under normalized=0, whereas
     * normalized=1 on the same weights multiplies by sum_h * sum_v = 24. */
    for (int i = 0; i < W * H; i++)
        img[i] = 3.0f;
    _iqa_convolve(img, W, H, &unnormalized, out, &rw, &rh);
    _iqa_convolve(img, W, H, &unscaled_weights, norm_out, &rw, &rh);
    for (int i = 0; i < rw * rh; i++) {
        mu_assert("constant image not preserved", almost_equal(out[i], 3.0f));
        mu_assert("negative control should be unscaled 24x",
                  !almost_equal(norm_out[i], 3.0f) &&
                  almost_equal(norm_out[i], 72.0f));
    }

    /* Rectangular asymmetric kernel (5x3) with negative vertical sum (-4). */
    float rect_h[5] = {1.0f, 2.0f, 3.0f, 2.0f, 0.0f};
    float rect_v[3] = {-1.0f, -2.0f, -1.0f};
    float rect_2d[15];
    build_2d_kernel(rect_h, 5, rect_v, 3, rect_2d);
    struct _kernel rect_k = {
        .kernel = rect_2d, .kernel_h = rect_h, .kernel_v = rect_v,
        .w = 5, .h = 3, .normalized = 0, .bnd_opt = KBND_SYMMETRIC,
    };

    fill_image(img);
    convolve_2d_ref(img, W, H, &rect_k, ref_2d, &ref_rw, &ref_rh);
    _iqa_convolve(img, W, H, &rect_k, out, &rw, &rh);
    mu_assert("unexpected 5x3 output width", rw == W - 4 && rw == ref_rw);
    mu_assert("unexpected 5x3 output height", rh == H - 2 && rh == ref_rh);
    for (int i = 0; i < rw * rh; i++)
        mu_assert("rectangular negative-sum separable disagrees with 2D control",
                  almost_equal(out[i], ref_2d[i]));

    return NULL;
}

/* If either 1D pass sums to zero, the equivalent 2D kernel sums to zero;
 * existing zero-sum behavior leaves the kernel unscaled (scale 1.0) on both
 * passes rather than scaling by the non-zero pass sum. */
static char *test_convolve_zero_sum_separable(void)
{
    float deriv[3] = {-1.0f, 0.0f, 1.0f};
    float smooth[3] = {1.0f, 4.0f, 1.0f};
    float k_hx_2d[9], k_vy_2d[9], k_xy_2d[9];
    float img[W * H], out[W * H], ref_2d[W * H], filt_2d[W * H];
    int rw = -1, rh = -1, ref_rw = -1, ref_rh = -1;

    build_2d_kernel(deriv, 3, smooth, 3, k_hx_2d);
    build_2d_kernel(smooth, 3, deriv, 3, k_vy_2d);
    build_2d_kernel(deriv, 3, deriv, 3, k_xy_2d);

    struct _kernel sobel_x = {
        .kernel = k_hx_2d, .kernel_h = deriv, .kernel_v = smooth,
        .w = 3, .h = 3, .normalized = 0, .bnd_opt = KBND_SYMMETRIC,
    };
    struct _kernel sobel_y = {
        .kernel = k_vy_2d, .kernel_h = smooth, .kernel_v = deriv,
        .w = 3, .h = 3, .normalized = 0, .bnd_opt = KBND_SYMMETRIC,
    };
    struct _kernel cross_xy = {
        .kernel = k_xy_2d, .kernel_h = deriv, .kernel_v = deriv,
        .w = 3, .h = 3, .normalized = 0, .bnd_opt = KBND_SYMMETRIC,
    };

    /* 1. Horizontal zero-sum, vertical non-zero sum (sum_h = 0, sum_v = 6). */
    fill_image(img);
    convolve_2d_ref(img, W, H, &sobel_x, ref_2d, &ref_rw, &ref_rh);
    _iqa_convolve(img, W, H, &sobel_x, out, &rw, &rh);
    int err = _iqa_img_filter(img, W, H, &sobel_x, filt_2d);
    mu_assert("_iqa_img_filter sobel_x failed", err == 0);
    for (int y = 0; y < rh; y++) {
        for (int x = 0; x < rw; x++) {
            const int idx = y * rw + x;
            mu_assert("horizontal zero-sum separable disagrees with 2D control",
                      almost_equal(out[idx], ref_2d[idx]));
            mu_assert("horizontal zero-sum separable disagrees with _iqa_img_filter",
                      almost_equal(out[idx], filt_2d[(y + 1) * W + (x + 1)]));
        }
    }

    /* Known horizontal ramp: f(x, y) = 2.5*x + 1 => dx = 5.0, smooth*dx = 30.0. */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y * W + x] = 2.5f * (float)x + 1.0f;
    _iqa_convolve(img, W, H, &sobel_x, out, &rw, &rh);
    for (int i = 0; i < rw * rh; i++)
        mu_assert("horizontal zero-sum ramp scaled unexpectedly",
                  almost_equal(out[i], 30.0f));

    /* Constant image under zero-sum filter must yield 0. */
    for (int i = 0; i < W * H; i++)
        img[i] = 7.0f;
    _iqa_convolve(img, W, H, &sobel_x, out, &rw, &rh);
    for (int i = 0; i < rw * rh; i++)
        mu_assert("zero-sum filter on constant image must be 0",
                  almost_equal(out[i], 0.0f));

    /* 2. Vertical zero-sum, horizontal non-zero sum (sum_h = 6, sum_v = 0). */
    fill_image(img);
    convolve_2d_ref(img, W, H, &sobel_y, ref_2d, &ref_rw, &ref_rh);
    _iqa_convolve(img, W, H, &sobel_y, out, &rw, &rh);
    err = _iqa_img_filter(img, W, H, &sobel_y, filt_2d);
    mu_assert("_iqa_img_filter sobel_y failed", err == 0);
    for (int y = 0; y < rh; y++) {
        for (int x = 0; x < rw; x++) {
            const int idx = y * rw + x;
            mu_assert("vertical zero-sum separable disagrees with 2D control",
                      almost_equal(out[idx], ref_2d[idx]));
            mu_assert("vertical zero-sum separable disagrees with _iqa_img_filter",
                      almost_equal(out[idx], filt_2d[(y + 1) * W + (x + 1)]));
        }
    }

    /* Known vertical ramp: f(x, y) = 1.5*y - 2 => smooth = 9*y - 12, dy = 18.0. */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y * W + x] = 1.5f * (float)y - 2.0f;
    _iqa_convolve(img, W, H, &sobel_y, out, &rw, &rh);
    for (int i = 0; i < rw * rh; i++)
        mu_assert("vertical zero-sum ramp scaled unexpectedly",
                  almost_equal(out[i], 18.0f));

    /* 3. Both horizontal and vertical zero-sum (sum_h = 0, sum_v = 0). */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y * W + x] = (float)(x * y);
    convolve_2d_ref(img, W, H, &cross_xy, ref_2d, &ref_rw, &ref_rh);
    _iqa_convolve(img, W, H, &cross_xy, out, &rw, &rh);
    for (int i = 0; i < rw * rh; i++)
        mu_assert("both zero-sum separable disagrees with 2D control (4.0)",
                  almost_equal(out[i], 4.0f) && almost_equal(out[i], ref_2d[i]));

    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_convolve_unnormalized_separable);
    mu_run_test(test_convolve_zero_sum_separable);
    return NULL;
}
