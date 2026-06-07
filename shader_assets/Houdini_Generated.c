#ifndef __INTERPOLATE_H__
#define __INTERPOLATE_H__

// #pragma OPENCL EXTENSION cl_amd_printf : enable

/*#ifdef cl_amd_printf
#define CHECKNAN(f) if (isnan(f)) { printf("%s is NAN, x = %g, y = %g, z = %g\n",#f, x, y, z);}
#define CHECKNAN2D(f) if (isnan(f)) { printf("%s is NAN, x = %g, y = %g\n",#f, x, y);}
#else*/
#define CHECKNAN(f)
#define CHECKNAN2D(f)
// #endif

#define CLAMPOFFSET 1e-4f

// While mix is limited to 0..1, it is tempting to just use a clamp
// on t; but you can get cancellation at t == 1 that stops the result
// from being b.   Selecting on t == 0 will also stop any NANs from
// leaking across, which matches the theory that a mix with 0 should
// able to be short-circuited with an if (t == 0) a, so the fact the
// second component is nan should not affect the result in this case.
#define _SAFE_MIX(a, b, t) \
    select( select( mix(a, b, t), a, t <= 0 ), b, t >= 1 )

static float safe_mix1(float a, float b, float t) { return _SAFE_MIX(a, b, t); }
static float2 safe_mix2(float2 a, float2 b, float2 t) { return _SAFE_MIX(a, b, t); }
static float3 safe_mix3(float3 a, float3 b, float3 t) { return _SAFE_MIX(a, b, t); }
static float4 safe_mix4(float4 a, float4 b, float4 t) { return _SAFE_MIX(a, b, t); }

#undef _SAFE_MIX

// The build methods are usually inferior to the direct cast,
// but ensure the result is an RValue so people can't assign to
// them in an unexpected manner.
static float2
build_float2(float x, float y)
{
    return (float2)(x, y);
}
static float3
build_float3(float x, float y, float z)
{
    return (float3)(x, y, z);
}
static float4
build_float4(float x, float y, float z, float w)
{
    return (float4)(x, y, z, w);
}

static float
lerp(float v1, float v2, float t)
{
    return v1 + (v2 - v1)*t;
}

static float
fit01(float t, float v1, float v2)
{
    return mix(v1, v2, clamp(t, 0.0f, 1.0f));
}

static float
fitTo01(float val, float omin, float omax)
{
    float d = omax - omin;
    if (fabs(d) < 1e-8f)
        return 0.5f;
    if (omin < omax)
    {
        if (val < omin) return 0;
        if (val > omax) return 1;
    }
    else
    {
        if (val < omax) return 1;
        if (val > omin) return 0;
    }
    return (val - omin) / d;
}

static float
fit(float val, float omin, float omax, float nmin, float nmax)
{
    return mix(nmin, nmax, fitTo01(val, omin, omax));
}

// in is an array size long of values to be interpolated within the
// [0, 1] interval.
// pos==0 => in[0]
// pos==1 => in[size-1]
static float
lerpConstant( constant float * in, int size, float pos )
{
    int m = size - 1;
    float flr;
    float t = fract(clamp(pos, 0.0f, 1.0f) * m, &flr);
    int flooridx = convert_int(flr);
    int ceilidx = min(flooridx+1, m);
    return mix(in[flooridx], in[ceilidx], t);
}

static float3
lerpConstant3( constant float * in, int size, float pos )
{
    int m = size - 1;
    float flr;
    float t = fract(clamp(pos, 0.0f, 1.0f) * m, &flr);
    int flooridx = convert_int(flr);
    int ceilidx = min(flooridx+1, m);
    float3 v1 = vload3(flooridx, in);
    float3 v2 = vload3(ceilidx, in);
    return mix(v1, v2, t);
}

static float
centerFromFace(__global const float *a, size_t idx, uint axisstride)
{
    return 0.5f * (a[idx] + a[idx + axisstride]);
}

static float
faceFromCenter(__global const float *a, size_t idx, uint axisstride)
{
    return 0.5f * (a[idx - axisstride] + a[idx]);
}

static float
cornerFromCenter(__global const float *a, size_t idx,
                 uint ystride, uint zstride)
{
    return 0.125f * (a[idx]
                   + a[idx - 1]
                   + a[idx - ystride]
                   + a[idx - zstride]
                   + a[idx - 1 - ystride]
                   + a[idx - 1 - zstride]
                   + a[idx - ystride - zstride]
                   + a[idx - 1 - ystride - zstride]);
}
static float
cornerFromCenter2d(__global const float *a, size_t idx,
                   uint xstride, uint ystride)
{
    return 0.25f * (a[idx] + a[idx - xstride] + a[idx - ystride]
                           + a[idx - xstride - ystride]);
}

// Calc central difference derivative of cell-centered grid
// at a center cell at idx.
static float
dudxAligned(__global const float *u, const uint idx,
                  const uint xstride, const float inv2dx)
{
    return inv2dx * (u[idx + xstride] - u[idx - xstride]);
}

// Calc central difference derivative of face-sampled grid
// at a center cell at idx.  This works for all the off-axes directions
// other than the one the face-sampled grid represents.  E.g. you can
// take the dy-derivative of the x-velocity field.
static float
dudxAlignedFace(__global const float *u, const uint idx,
                      const uint ustride, const uint xstride, float inv4dx)
{
    return inv4dx * ((u[idx + xstride] + u[idx + ustride + xstride]) -
                     (u[idx - xstride] + u[idx + ustride - xstride]));
}

// Calc central difference derivative of face-sampled grid
// at a center cell at idx. This only works for the derivative along
// the axis the face-centered grid represents, e.g. you can only take
// the dx-derivative of the x-velocity field.
static float
dudxFaceAtCenter(__global const float *u, const uint idx,
                      const uint xstride, float invdx)
{
    return invdx * (u[idx + xstride] - u[idx]);
}

static float
dudxCenterAtFace(__global const float *u, const uint idx,
                      const uint xstride, float invdx)
{
    return invdx * (u[idx] - u[idx - xstride]);
}

// Calc central difference derivative of center-sampled grid at a corner cell
// at idx, by first averaging along relevant faces (invdx should incorporate
// the averaging factor). axis_stride is stride along the differentiation axis,
// and the other strides are along the remaining axes.
static float
dudxCenterAtCorner(__global const float *u, const uint idx,
                   const uint axis_stride, const uint off_stride1,
                   const uint off_stride2, float inv4dx)
{
    return inv4dx * ((u[idx] + u[idx - off_stride1]
            + u[idx - off_stride2] + u[idx - off_stride1 - off_stride2])
                 -  (u[idx - axis_stride] + u[idx - off_stride1 - axis_stride]
            + u[idx - off_stride2 - axis_stride]
            + u[idx - off_stride1 - off_stride2 - axis_stride]));
}

static float
dudxCenterAtCorner2d(__global const float *u, const uint idx,
                     const uint axis_stride, const uint off_stride,
                     float inv2dx)
{
    return inv2dx * ((u[idx] + u[idx - off_stride])
                 -  (u[idx - axis_stride] + u[idx - off_stride - axis_stride]));
}

static void
bilinear_interp(float x, float y, __global const float *p,
                        size_t idx,
                        __global float *phin,
                        __global float *minphi,
                        __global float *maxphi,
                        uint offset, uint xstride, uint ystride)

{
    // clamp to boundaries
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);

    // get fractional part
    const float sx = x - gi;
    const float sy = y - gj;

    size_t srcidx = offset + gi * xstride + gj * ystride;

    const float i00 = p[srcidx];
    const float i10 = p[srcidx + xstride];
    const float i01 = p[srcidx + ystride];
    const float i11 = p[srcidx + xstride + ystride];

    CHECKNAN2D(i00)
    CHECKNAN2D(i10)
    CHECKNAN2D(i01)
    CHECKNAN2D(i11)
    const float val = (i00 * (1-sx) + i10 * (sx)) * (1-sy) +
                      (i01 * (1-sx) + i11 * (sx)) * (  sy);
    phin[idx] = val;
    if (minphi)
        minphi[idx] = fmin(fmin(fmin(i00, i01), i10), i11);
    if (maxphi)
        maxphi[idx] = fmax(fmax(fmax(i00, i01), i10), i11);
}

static float
bilinear_interp_val(float x, float y, __global const float *p,
                        uint offset, uint xstride, uint ystride,
                        uint offx,  uint offy)
{
    // clamp to boundaries
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET - offx);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET - offy);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);

    // get fractional part
    const float sx = x - gi;
    const float sy = y - gj;

    size_t srcidx = offset + gi * xstride + gj * ystride;

    const float i00 = p[srcidx];
    const float i10 = p[srcidx + xstride];
    const float i01 = p[srcidx + ystride];
    const float i11 = p[srcidx + xstride + ystride];

    return (i00 * (1-sx) + i10 * (sx)) * (1-sy) +
           (i01 * (1-sx) + i11 * (sx)) * (  sy);
}

static void
trilinear_interp(float x, float y, float z, __global const float *p,
                    size_t idx,
                    __global float *phin,
                    __global float *minphi,
                    __global float *maxphi,
                    uint offset, uint ystride, uint zstride)
{
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET);
    z = clamp(z, -1.0f, get_global_size(2) - CLAMPOFFSET);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);
    const int gk = (int)floor(z);

    const float sx = x - gi;
    const float sy = y - gj;
    const float sz = z - gk;

    size_t srcidx = offset + gi + gj * ystride + gk * zstride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + 1];
    const float i010 = p[srcidx + ystride];
    const float i110 = p[srcidx + 1 + ystride];
    const float i001 = p[srcidx + zstride];
    const float i101 = p[srcidx + 1 + zstride];
    const float i011 = p[srcidx + ystride + zstride];
    const float i111 = p[srcidx + 1 + ystride + zstride];

    CHECKNAN(i000)
    CHECKNAN(i100)
    CHECKNAN(i010)
    CHECKNAN(i110)
    CHECKNAN(i001)
    CHECKNAN(i101)
    CHECKNAN(i011)
    CHECKNAN(i111)
    const float val = ((i000 * (1 - sx) + i100 * (sx)) * (1 - sy) +
                      (i010 * (1 - sx) + i110 * (sx)) * (  sy)) * (1 - sz) +
                      ((i001 * (1 - sx) + i101 * (sx)) * (1 - sy) +
                      (i011 * (1 - sx) + i111 * (sx)) * (  sy)) * (  sz);

    phin[idx] = val;
    if (minphi)
        minphi[idx] = fmin(fmin(fmin(fmin(fmin(fmin(fmin(i000, i001), i010),
                                            i011), i100), i101), i110), i111);
    if (maxphi)
        maxphi[idx] = fmax(fmax(fmax(fmax(fmax(fmax(fmax(i000, i001), i010),
                                            i011), i100), i101), i110), i111);
}

static float
trilinear_interp_val(float x, float y, float z, __global const float *p,
                    uint offset, uint ystride, uint zstride,
                    uint offx,  uint offy, uint offz )
{
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET - offx);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET - offy);
    z = clamp(z, -1.0f, get_global_size(2) - CLAMPOFFSET - offz);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);
    const int gk = (int)floor(z);

    const float sx = x - gi;
    const float sy = y - gj;
    const float sz = z - gk;

    size_t srcidx = offset + gi + gj * ystride + gk * zstride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + 1];
    const float i010 = p[srcidx + ystride];
    const float i110 = p[srcidx + 1 + ystride];
    const float i001 = p[srcidx + zstride];
    const float i101 = p[srcidx + 1 + zstride];
    const float i011 = p[srcidx + ystride + zstride];
    const float i111 = p[srcidx + 1 + ystride + zstride];

    return  ((i000 * (1 - sx) + i100 * (sx)) * (1 - sy) +
            (i010 * (1 - sx) + i110 * (sx)) * (  sy)) * (1 - sz) +
            ((i001 * (1 - sx) + i101 * (sx)) * (1 - sy) +
            (i011 * (1 - sx) + i111 * (sx)) * (  sy)) * (  sz);
}

static float
bilinear_interp_vol(float2 pos, __global const float *p,
                    uint offset, uint xstride, uint ystride,
                    uint resx, uint resy)
{
    float x = clamp(pos.x-0.5f, (float)0, (float)(resx-1));
    float y = clamp(pos.y-0.5f, (float)0, (float)(resy-1));

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);

    // In case our clamp is exactly res#-1
    const int dx = select((int)xstride, (int)0, gi == (int)(resx)-1);
    const int dy = select((int)ystride, (int)0, gj == (int)(resy)-1);

    const float sx = x - gi;
    const float sy = y - gj;

    size_t srcidx = offset + gi * xstride + gj * ystride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + dx];
    const float i010 = p[srcidx + dy];
    const float i110 = p[srcidx + dx + dy];

    return mix( mix(i000, i100, sx),
                mix(i010, i110, sx), sy);
}

static float
trilinear_interp_vol(float3 pos, __global const float *p,
                    uint offset, uint xstride, uint ystride, uint zstride,
                    uint resx, uint resy, uint resz)
{
    float x = clamp(pos.x-0.5f, (float)0, (float)(resx-1));
    float y = clamp(pos.y-0.5f, (float)0, (float)(resy-1));
    float z = clamp(pos.z-0.5f, (float)0, (float)(resz-1));

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);
    const int gk = (int)floor(z);

    // In case our clamp is exactly res#-1
    const int dx = select((int)xstride, (int)0, gi == (int)(resx)-1);
    const int dy = select((int)ystride, (int)0, gj == (int)(resy)-1);
    const int dz = select((int)zstride, (int)0, gk == (int)(resz)-1);

    const float sx = x - gi;
    const float sy = y - gj;
    const float sz = z - gk;

    size_t srcidx = offset + gi * xstride + gj * ystride + gk * zstride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + dx];
    const float i010 = p[srcidx + dy];
    const float i110 = p[srcidx + dx + dy];
    const float i001 = p[srcidx + dz];
    const float i101 = p[srcidx + dx + dz];
    const float i011 = p[srcidx + dy + dz];
    const float i111 = p[srcidx + dx + dy + dz];

    return mix( mix( mix(i000, i100, sx),
                     mix(i010, i110, sx), sy),
                mix( mix(i001, i101, sx),
                     mix(i011, i111, sx), sy),
                sz);
}

static float4
linesegInterpolationWeights(float u)
{
    return (float4)(1 - u, u, 0, 0);
}

// From GEOtriInterpolationWeights, but
static float4
triInterpolationWeights(float u, float v)
{
    // Triangle - use barycentric coordinates

    // This is a hack to make sure we are given proper
    // barycentric u, v coordinates.  That is, we require
    // u+v <= 1, and if that's not the case we hack it so
    // u = 1-u, v = 1-v, thus ensuring u+v <= 1.  (This
    // assumes that u, v are each between 0 and 1)
    // This is used for when evaluateInteriorPoint is
    // called from POP_GenVar for birthing from a surface.
    //
    // Note we actually flip on the u+v = 1 axis instead
    // of what is described above so slightly outside points
    // do not teleport to opposite locations.
    float uv = 1 - u - v;

    // Assume valid uv's.
#if 0
    if (uv < 0)
    {
        u += uv;
        v += uv;
        uv = -uv;
    }
#endif

    return (float4) (uv, u, v, 0);
}

// From GEOquadInterpolationWeights
static float4
quadInterpolationWeights(float u, float v)
{
    float u1 = 1 - u;
    float v1 = 1 - v;
    return (float4)(u1 * v1, u1 * v, u * v, u * v1);
}

// From GEO_PrimTetrahedron::remapTetCoords
static float4
tetInterpolationWeights(float u, float v, float w)
{
    float uvw = 1 - u - v - w;

    // Assume valid uv's.
#if 0
    if (uvw < 0)
    {
        // Mirror in u + v == 1 plane, reducing from 6 tetrahedra to 3,
        // i.e. a right triangular prism, whose right-angle edge is
        // along the w axis.
        if (u + v > 1)
        {
            float bary = 1 - u - v;
            u += bary;
            v += bary;
            uvw -= 2 * bary;
        }
        // Mirror the far tetrahedron (shares no face with the tet to keep)
        // into the middle tetrahedron (shares one face with the tet to keep)
        if (u + w > 1)
        {
            // Weight of point at (1,0,1), since 1+1-1 = 1
            float weight = (u + w - 1);
            // Subtract component of (1,0,1), which only changes u and w
            u -= weight;//*1
            //v -= weight*0;
            w -= weight;//*1
            // Add component of (0,1,0), which only changes v
            //u += weight*0;
            v += weight;//*1
            //w += weight*0;
            // Update uvw
            uvw += weight;
        }
        // Mirror the remaining outside tetrahedron into the tet to keep
        if (uvw < 0)
        {
            // Weight of point at (0,1,1), since -(1-0-1-1) = 1
            float weight = -uvw;
            // Subtract component of (0,1,1), which only changes v and w
            //u -= weight*0
            v -= weight;//*1
            w -= weight;//*1
            // Add component of (0,0,0), which requires no change
            // Update uvw
            uvw = -uvw; //equivalent to uvw += 2*weight;
        }
    }
#endif

    return (float4)(uvw, u, v, w);
}

static void
computeSubdCurveCoeffsAndIndices(
    float u,
    int n,
    bool closed,
    float4 *coeffs_ptr,
    int4 *indices_ptr)
{
    const float16 theSubDFirstBasis = {
        1.0, -1.0,  0.0,  1.0/6.0,
        0.0,  1.0,  0.0, -1.0/3.0,
        0.0,  0.0,  0.0,  1.0/6.0,
        0.0,  0.0,  0.0,  0.0
    };

    const float16 theOpenBasis = {
        1.0/6.0, -0.5,  0.5, -1.0/6.0,
        2.0/3.0,  0.0, -1.0,  0.5,
        1.0/6.0,  0.5,  0.5, -0.5,
        0.0,      0.0,  0.0,  1.0/6.0
    };

    float4 coeffs = 0.0;
    int4 indices = 0.0;

    // UT_ASSERT_P(n >= 1);

    // Special cases for n <= 2
    if (n == 1)
    {
        indices.x = -1;
        indices.y = 0;
        indices.z = -1;
        indices.w = -1;

        coeffs.x = 0;
        coeffs.y = 1;
        coeffs.z = 0;
        coeffs.w = 0;

        *coeffs_ptr = coeffs;
        *indices_ptr = indices;
    }
    else if (n == 2)
    {
        if (closed)
        {
            u *= 2;
            if (u > 1)
                u = 2-u;
        }

        indices.x = -1;
        indices.y = 0;
        indices.z = 1;
        indices.w = -1;

        coeffs.x = 0.0;
        coeffs.y = 1.0 - u;
        coeffs.z = u;
        coeffs.w = 0.0;
    }

    const float16 *basis = &theOpenBasis;
    // const float16 *midbasis = &theOpenBasis;

    int nedges = n - !closed;
    u *= nedges;
    int i = floor(u);
    i = clamp(i, 0, nedges-1);
    float t = u - i;

    float16 temp;
    if (i == 0)
    {
        if (closed)
        {
            indices.x = n-1;
            indices.y = 0;
            indices.z = 1;
            indices.w = 2;
        }
        else
        {
            basis = &theSubDFirstBasis;

            indices.x = 0;
            indices.y = 1;
            indices.z = 2;
            indices.w = -1;
        }
    }
    else if (i >= n-2)
    {
        if (closed)
        {
            if (i == n-2)
            {
                indices.x = n-3;
                indices.y = n-2;
                indices.z = n-1;
                indices.w = 0;
            }
            else
            {
                // UT_ASSERT_P(i == n-1);
                indices.x = n-2;
                indices.y = n-1;
                indices.z = 0;
                indices.w = 1;
            }
        }
        else
        {
            basis = &theSubDFirstBasis;

            indices.x = -1;
            indices.y = n-3;
            indices.z = n-2;
            indices.w = n-1;

            temp.lo.lo = (*basis).hi.hi;
            temp.lo.hi = (*basis).hi.lo;
            temp.hi.lo = (*basis).lo.hi;
            temp.hi.hi = (*basis).lo.lo;
            basis = &temp;
            t = 1-t;
        }
    }
    else
    {
        indices.x = i-1;
        indices.y = i;
        indices.z = i+1;
        indices.w = i+2;
    }

    float t2 = t*t;
    float4 tpow = {1.0f, t, t2, t2*t};

    (*coeffs_ptr).x = dot(tpow, (*basis).lo.lo);
    (*coeffs_ptr).y = dot(tpow, (*basis).lo.hi);
    (*coeffs_ptr).z = dot(tpow, (*basis).hi.lo);
    (*coeffs_ptr).w = dot(tpow, (*basis).hi.hi);

    *indices_ptr = indices;
}

static float3 
evaluateCubicCoeffs_vload3(float4 coeffs, int4 indices, global float *values)
{
    float3 res = 0;

    if (indices.x >= 0)
        res += coeffs.x * vload3(indices.x, values);
    if (indices.y >= 0)
        res += coeffs.y * vload3(indices.y, values);
    if (indices.z >= 0)
        res += coeffs.z * vload3(indices.z, values);
    if (indices.w >= 0)
        res += coeffs.w * vload3(indices.w, values);

    return res;
}

static float3 
evaluateCubicCoeffs_3(float4 coeffs, int4 indices, float3 *values)
{
    float3 res = 0;

    if (indices.x >= 0)
        res += coeffs.x * values[indices.x];
    if (indices.y >= 0)
        res += coeffs.y * values[indices.y];
    if (indices.z >= 0)
        res += coeffs.z * values[indices.z];
    if (indices.w >= 0)
        res += coeffs.w * values[indices.w];

    return res;
}
#endif
/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    typedefines.h ( CE Library, OpenCL)
 *
 * COMMENTS: OpenCL type definitions
 */


#ifndef __TYPE_DEFINE_H__
#define __TYPE_DEFINE_H__

// The OpenCL SOP/DOP might define
// these before compilation for 32-/64-bit support,
// so only define them if not already defined.
#ifndef fpreal
// The USE_DOUBLE flag is used by the older OpenCL-enabled
// DOPs that make up the Pyro solver.
#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define fpreal double
#define fpreal2 double2
#define fpreal3 double3
#define fpreal4 double4
#define fpreal8 double8
#define fpreal16 double16
#define FPREAL_PREC 64
// We also want to define exint as long in this case.
#define USE_LONG

#else
#define fpreal float
#define fpreal2 float2
#define fpreal3 float3
#define fpreal4 float4
#define fpreal8 float8
#define fpreal16 float16
#define FPREAL_PREC 32

#endif
#endif


#if FPREAL_PREC==64

// Load a 64-bit fpreal2 from a float2 buffer.
static fpreal2
vload2f(size_t i, const global float *b)
{
    i *= 2;
    return (fpreal2)(b[i], b[i + 1]);
}

// Load a 64-bit fpreal3 from a float3 buffer.
static fpreal3
vload3f(size_t i, const global float *b)
{
    i *= 3;
    return (fpreal3)(b[i], b[i+1], b[i+2]);
}

// Load a 64-bit fpreal4 from a float4 buffer.
static fpreal4
vload4f(size_t i, const global float *b)
{
    i *= 4;
    return (fpreal4)(b[i], b[i+1], b[i+2], b[i+3]);
}

// Store a 64-bit fpreal3 into a float3 buffer.
static void
vstore3f(fpreal3 a, size_t i, global float *b)
{
    vstore3((float3)(a.x, a.y, a.z), i, b);
}

// Store a 64-bit fpreal4 into a float4 buffer.
static void
vstore4f(fpreal4 a, size_t i, global float *b)
{
    vstore4((float4)(a.x, a.y, a.z, a.w), i, b);
}

// Convert float2 to 64-bit fpreal2
static fpreal2
asfpreal2(float2 a)
{
    return (fpreal2)(a.x, a.y);
}

// Convert float3 to 64-bit fpreal3
static fpreal3
asfpreal3(float3 a)
{
    return (fpreal3)(a.x, a.y, a.z);
}

// Convert float4 to 64-bit fpreal4
static fpreal4
asfpreal4(float4 a)
{
    return (fpreal4)(a.x, a.y, a.z, a.w);
}

#else

// Load a 32-bit fpreal2 from a float2 buffer (no-op).
#define vload2f vload2

// Load a 32-bit fpreal3 from a float3 buffer (no-op).
#define vload3f vload3

// Load a 32-bit fpreal4 from a float4 buffer (no-op).
#define vload4f vload4

// Store a 32-bit fpreal3 into a float3 buffer (no-op).
#define vstore3f vstore3

// Store a 32-bit fpreal4 into a float4 buffer (no-op).
#define vstore4f vstore4

// Convert float2 to 32-bit fpreal2 (no-op)
#define asfpreal2(x) (x)

// Convert float3 to 32-bit fpreal3 (no-op)
#define asfpreal3(x) (x)

// Convert float4 to 32-bit fpreal4 (no-op)
#define asfpreal4(x) (x)

#endif

// The OpenCL SOP/DOP might define
// these before compilation for 32-/64-bit support,
// so only define them if not already defined.
#ifndef exint
#ifdef USE_LONG
#define exint  long
#define exint2 long2
#define exint3 long3
#define exint4 long4
#else
#define exint  int
#define exint2 int2
#define exint3 int3
#define exint4 int4
#endif
#endif

#endif
/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    util.h ( CE Library, OpenCL)
 *
 * COMMENTS:
 */

#ifndef __UTIL_H__
#define __UTIL_H__

static void swapf(fpreal *a, fpreal *b)
{
    fpreal t = *a;
    *a = *b;
    *b = t;
}

#endif
/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    matrix.h ( CE Library, OpenCL)
 *
 * COMMENTS:
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

// #include "typedefines.h"
// #include "util.h"

#define PRINTI(v)                                                              \
    printf("%s:\n", #v);                                                       \
    printf("%d\n", v)

#define PRINTU(v)                                                              \
    printf("%s:\n", #v);                                                       \
    printf("%u\n", v)

#define PRINTLI(v)                                                             \
    printf("%s:\n", #v);                                                       \
    printf("%lld\n", v)

#define PRINTLU(v)                                                             \
    printf("%s:\n", #v);                                                       \
    printf("%llu\n", v)

#define PRINTF(v)                                                              \
    printf("%s:\n", #v);                                                       \
    printf("%g\n", v)

#define PRINTVEC3(v)                                                           \
    printf("%s:\n", #v);                                                       \
    printf("%g %g %g\n", v.x, v.y, v.z)

#define PRINTVEC3I(v)                                                          \
    printf("%s:\n", #v);                                                       \
    printf("%d %d %d\n", v.x, v.y, v.z)

#define PRINTMAT3(m)                                                           \
    printf("%s:\n", #m);                                                       \
    printf("%g %g %g\n", m[0].s0, m[0].s1, m[0].s2);                           \
    printf("%g %g %g\n", m[1].s0, m[1].s1, m[1].s2);                           \
    printf("%g %g %g\n", m[2].s0, m[2].s1, m[2].s2)

#define PRINTMAT3NP(m)                                                         \
    printf("%s = np.array((\n", #m);                                         \
    printf("\t(%g, %g, %g),\n", m[0].s0, m[0].s1, m[0].s2);                    \
    printf("\t(%g, %g, %g),\n", m[1].s0, m[1].s1, m[1].s2);                    \
    printf("\t(%g, %g, %g)))\n", m[2].s0, m[2].s1, m[2].s2)

#define PRINTMAT4(m)                                                           \
    printf("%s:\n", #m);                                                       \
    printf("%g %g %g %g\n", m.s0, m.s1, m.s2, m.s3);                           \
    printf("%g %g %g %g\n", m.s4, m.s5, m.s6, m.s7);                           \
    printf("%g %g %g %g\n", m.s8, m.s9, m.sa, m.sb);                           \
    printf("%g %g %g %g\n", m.sc, m.sd, m.se, m.sf)

// A 3x3 matrix in row-major order (to match UT_Matrix3)
// NOTE: fpreal3 is 4 floats, so this is size 12
typedef fpreal3 mat3[3];  

// A 3x2 matrix in row-major order
typedef fpreal2 mat32[3];

// A 2x2 matrix in row-major order, stored in a single fpreal4
typedef fpreal4 mat2;

// A 4x4 matrix in row-major order, stored in a single fpreal16
typedef fpreal16 mat4;

// Return the sum of entries of the vector v.
static fpreal
vec3sum(const fpreal3 v)
{
    return v.x + v.y + v.z;
}

// Return the product of entries of the vector v.
static fpreal
vec3prod(const fpreal3 v)
{
    return v.x * v.y * v.z;
}

// Create a 2x2 matrix with columns of the specified vectors.
static mat2
mat2fromcols(const fpreal2 c0, const fpreal2 c1)
{
    return (mat2)(c0.s0, c1.s0,
                  c0.s1, c1.s1);
}

// Transpose a 2x2 matrix.
static mat2
transpose2(const mat2 a)
{
    return (mat2)(a.even, a.odd);
}

// Multiply A * B for 2x2 matrices.
static mat2
mat2mul(const mat2 a, const mat2 b)
{
    return (mat2)(dot(a.lo, b.even), dot(a.lo, b.odd),
                  dot(a.hi, b.even), dot(a.hi, b.odd));
}

// Multiply b * A where b is a 2-vector and A is a 2x2 matrix.
// This multiplication order matches VEX.
static fpreal2
mat2vecmul(const mat2 a, const fpreal2 b)
{
    mat2 aT = transpose2(a);
    return (fpreal2)(dot(aT.lo, b), dot(aT.hi, b));
}

// Return the square of the L2-norm of a 2x2 matrix.
static fpreal
squaredNorm2(const mat2 a)
{
    return dot(a.lo, a.lo) + dot(a.hi, a.hi);
}

// Add 3x3 matrix A to matrix B and store the result in matrix C.
static void
mat3add(const mat3 a, const mat3 b, mat3 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}

// Subtract 3x3 matrix A from matrix B and store the result in C.
static void
mat3sub(const mat3 a, const mat3 b, mat3 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

// Set 3x3 matrix A to zero.
static void
mat3zero(mat3 a)
{
    a[0] = (fpreal3)(0.0f);
    a[1] = (fpreal3)(0.0f);
    a[2] = (fpreal3)(0.0f);
}

// Set 3x3 matrix A to the identity.
static void
mat3identity(mat3 a)
{
    a[0] = (fpreal3)(1.0f, 0.0f, 0.0f);
    a[1] = (fpreal3)(0.0f, 1.0f, 0.0f);
    a[2] = (fpreal3)(0.0f, 0.0f, 1.0f);
}

// Copy 3x3 matrix A to matrix B.
static void
mat3copy(const mat3 a, mat3 b)
{
    b[0] = a[0];
    b[1] = a[1];
    b[2] = a[2];
}

// Load a 3x3 matrix from memory at the specified index.
// The matrix should be row-major as stored in geometry
// attributes.
static void
mat3load(size_t idx, const global float *a, mat3 m)
{
    idx *= 3;
    m[0] = vload3f(idx, a);
    m[1] = vload3f(idx + 1, a);
    m[2] = vload3f(idx + 2, a);
}

// Store a 3x3 matrix to memory at the specified index.
// The matrix should be row-major as stored in geometry
// attributes.
static void 
mat3store(mat3 in, int idx, global fpreal *data)
{
    idx *= 3;
    vstore3(in[0], idx, data);
    vstore3(in[1], idx + 1, data);
    vstore3(in[2], idx + 2, data);
}

// Create a 3x3 matrix with columns of the specified vectors.
static void
mat3fromcols(const fpreal3 c0, const fpreal3 c1, const fpreal3 c2, mat3 m)
{
    m[0] = (fpreal3)(c0.s0, c1.s0, c2.s0);
    m[1] = (fpreal3)(c0.s1, c1.s1, c2.s1);
    m[2] = (fpreal3)(c0.s2, c1.s2, c2.s2);
}

// Transpose a 3x3 matrix.
static void
transpose3(const mat3 a, mat3 b)
{
    mat3fromcols(a[0], a[1], a[2], b);
}

// Multiply A * B and store in C for 3x3 matrices.
static void
mat3mul(const mat3 a, const mat3 b, mat3 c)
{
    mat3 bT;
    transpose3(b, bT);
    c[0] = (fpreal3)(dot(a[0], bT[0]), dot(a[0], bT[1]), dot(a[0], bT[2]));
    c[1] = (fpreal3)(dot(a[1], bT[0]), dot(a[1], bT[1]), dot(a[1], bT[2]));
    c[2] = (fpreal3)(dot(a[2], bT[0]), dot(a[2], bT[1]), dot(a[2], bT[2]));
}

// Multiply A * B^T and store in C for 3x3 matrices.
static void
mat3mulT(const mat3 a, const mat3 b, mat3 c)
{
    c[0] = (fpreal3)(dot(a[0], b[0]), dot(a[0], b[1]), dot(a[0], b[2]));
    c[1] = (fpreal3)(dot(a[1], b[0]), dot(a[1], b[1]), dot(a[1], b[2]));
    c[2] = (fpreal3)(dot(a[2], b[0]), dot(a[2], b[1]), dot(a[2], b[2]));
}

// Multiply b * A where b is a 3-vector and A is a 3x3 matrix.
// This multiplication order matches VEX.
static fpreal3
mat3vecmul(const mat3 a, const fpreal3 b)
{
    mat3 aT;
    transpose3(a, aT);
    return (fpreal3)(dot(aT[0], b), dot(aT[1], b), dot(aT[2], b));
}

// Multiply b * A^T where b is a 3-vector and A is a 3x3 matrix.
// This multiplication order matches VEX.
static fpreal3
mat3Tvecmul(const mat3 a, const fpreal3 b)
{
    return (fpreal3)(dot(a[0], b), dot(a[1], b), dot(a[2], b));
}

// Multiply b * A where b is a 3-vector and A is a 3x3 matrix,
// but discard the third component.
// This multiplication order matches VEX.
static fpreal2
mat3vec2mul(const mat3 a, const fpreal3 b)
{
    mat3 aT;
    transpose3(a, aT);
    return (fpreal2)(dot(aT[0], b), dot(aT[1], b));
}

// Multiply b * A^T where b is a 3-vector and A is a 3x3 matrix,
// but discard the third component.
// This multiplication order matches VEX.
static fpreal2
mat3Tvec2mul(const mat3 a, const fpreal3 b)
{
    return (fpreal2)(dot(a[0], b), dot(a[1], b));
}

// Store the 3x3 matrix that is the outer product of the input
// a and b vectors in C.
static void
outerprod3(const fpreal3 a, const fpreal3 b, mat3 c)
{
    c[0] = a.x * b;
    c[1] = a.y * b;
    c[2] = a.z * b;
}

// Compute C = s * A + t * B, where s, t are scalars and A, B are 3x3 matrices.
static void
mat3lcombine(const fpreal s, const mat3 a, const fpreal t, const mat3 b, mat3 c)
{
    c[0] = s * a[0] + t * b[0];
    c[1] = s * a[1] + t * b[1];
    c[2] = s * a[2] + t * b[2];
}

// Return the square of the L2-norm of a 3x3 matrix.
static fpreal
squaredNorm3(const mat3 a)
{
    return dot(a[0], a[0]) + dot(a[1], a[1]) + dot(a[2], a[2]);
}

// Return the determinant of the supplied 3x3 matrix.
static fpreal
det3(const mat3 a)
{
    fpreal d = a[0].s0 * (a[1].s1 * a[2].s2 - a[1].s2 * a[2].s1);
    d -= a[0].s1 * (a[1].s0 * a[2].s2 - a[1].s2 * a[2].s0);
    d += a[0].s2 * (a[1].s0 * a[2].s1 - a[1].s1 * a[2].s0);
    return d;
}

// Return diagonal vector of the supplied 3x3 matrix.
static fpreal3
diag3(const mat3 a)
{
    return (fpreal3)(a[0].s0, a[1].s1, a[2].s2);
}

// Set a to the 3x3 diagonal matrix defined by the entries of the vector diag.
static void
mat3diag(const fpreal3 diag, mat3 a)
{
    mat3zero(a);
    a[0].x = diag.x;
    a[1].y = diag.y;
    a[2].z = diag.z;
}

// Return the trace of the supplied 3x3 matrix.
static fpreal
trace3(const mat3 m)
{
    return vec3sum(diag3(m));
}

// Set 4x4 matrix A to the identity.
static void
mat4identity(mat4 *a)
{
    (*a).lo.lo = (fpreal4)(1.0f, 0.0f, 0.0f, 0.0f);
    (*a).lo.hi = (fpreal4)(0.0f, 1.0f, 0.0f, 0.0f);
    (*a).hi.lo = (fpreal4)(0.0f, 0.0f, 1.0f, 0.0f);
    (*a).hi.hi = (fpreal4)(0.0f, 0.0f, 0.0f, 1.0f);
}

// Multiply b * A where b is a 2-vector and A is a 4x4 matrix.
// This multiplication order matches VEX.
static fpreal2
mat4vec2mul(const mat4 a, const fpreal2 b)
{
    return b.x * a.lo.lo.xy +
           b.y * a.lo.hi.xy +
                 a.hi.hi.xy;
}

// Multiply b * A where b is a 3-vector and A is a 4x3 matrix,
// assuming a fourth component of the vector to be 0, i.e.
// the typical transformation of a 3d vector by a matrix.
// This multiplication order matches VEX.
static fpreal3
mat43vec3mul(const mat4 a, const fpreal3 b)
{
    fpreal4 result = b.x * a.lo.lo +
                    b.y * a.lo.hi +
                    b.z * a.hi.lo;
    return (fpreal3)(result.x, result.y, result.z);
}

// Multiply b * A where b is a 3-vector and A is a 4x4 matrix,
// assuming a fourth component of the vector to be 1, i.e.
// the typical transformation of a 3d point by a matrix.
// This multiplication order matches VEX.static fpreal3
static fpreal3
mat4vec3mul(const mat4 a, const fpreal3 b)
{
    fpreal4 result = b.x * a.lo.lo +
                    b.y * a.lo.hi +
                    b.z * a.hi.lo +
                    a.hi.hi;
    return (fpreal3)(result.x, result.y, result.z);
}

// Multiply b * A where b is a 4-vector and A is a 4x4 matrix.
// This multiplication order matches VEX.
static fpreal4
mat4vecmul(const mat4 a, const fpreal4 b)
{
    fpreal4 result = b.x * a.lo.lo +
                    b.y * a.lo.hi +
                    b.z * a.hi.lo +
                    b.w * a.hi.hi;
    return result;
}

// UT_Matrix4::solveColumn
// cp:          pivot_col and pivot_row
// c1, c2, c3:  other columns and rows
static void
mat4solvecol(fpreal (*matx)[4][4], int cp)
{
    fpreal      pivot_value_inverse;
    int         c1, c2, c3;

    switch (cp)
    {
        case 0: c1 = 1; c2 = 2; c3 = 3; break;
        case 1: c1 = 0; c2 = 2; c3 = 3; break;
        case 2: c1 = 0; c2 = 1; c3 = 3; break;
        case 3: c1 = 0; c2 = 1; c3 = 2; break;
    }

    // Here we will find the inverse of the pivot, set the pivot to 1,
    // and multiply the row which contains the pivot by the pivot
    // inverse. This might seem a little weird. The algorithm does it
    // this way so that it can gradually replace the input matrix
    // with its inverse.
    pivot_value_inverse = 1.0F/(*matx)[cp][cp];

    (*matx)[cp][cp] = pivot_value_inverse;
    (*matx)[cp][c1] *= pivot_value_inverse;
    (*matx)[cp][c2] *= pivot_value_inverse;
    (*matx)[cp][c3] *= pivot_value_inverse;

    // Now we subtract multiples of the pivot row from the other
    // rows in the matrix. This would be more familiar if the pivot
    // itself hadn't been set to 1 before the pivot row was multiplied
    // by the inverse (see above).

    (*matx)[c1][c1] -= (*matx)[cp][c1]*(*matx)[c1][cp];
    (*matx)[c1][c2] -= (*matx)[cp][c2]*(*matx)[c1][cp];
    (*matx)[c1][c3] -= (*matx)[cp][c3]*(*matx)[c1][cp];

    (*matx)[c2][c1] -= (*matx)[cp][c1]*(*matx)[c2][cp];
    (*matx)[c2][c2] -= (*matx)[cp][c2]*(*matx)[c2][cp];
    (*matx)[c2][c3] -= (*matx)[cp][c3]*(*matx)[c2][cp];

    (*matx)[c3][c1] -= (*matx)[cp][c1]*(*matx)[c3][cp];
    (*matx)[c3][c2] -= (*matx)[cp][c2]*(*matx)[c3][cp];
    (*matx)[c3][c3] -= (*matx)[cp][c3]*(*matx)[c3][cp];

    (*matx)[c1][cp] *= -pivot_value_inverse;
    (*matx)[c2][cp] *= -pivot_value_inverse;
    (*matx)[c3][cp] *= -pivot_value_inverse;
}

// UT_Matrix4::invert
// Linear equation solution by Gauss-Jordan elimination.
static int
mat4invert(fpreal16 *m)
{
    fpreal tol = 0.0;
    fpreal (*matx)[4][4] = (fpreal (*)[4][4]) m;
    
    int indexcol[4], indexrow[4];
    int pivot_row, pivot_col;

    // Check for the very common case of trivial column 3.
    const bool is_trivial_col3 =
        ((*matx)[0][3] == 0) &&
        ((*matx)[1][3] == 0) &&
        ((*matx)[2][3] == 0) &&
        ((*matx)[3][3] == 1);

    // We will need to keep track of which columns we have already found
    // pivots in. For this we use what is essentially a set of flags in the
    // array "pivoted_columns_flags". We initialize them to 0 here.
    bool pivoted_columns_flags[4] = {false, false, false, false};

    // In order to invert an n*n matrix, we will have to find n pivots, and
    // for each pivot reduce each row. We will keep track of which pivot and
    // set of reductions we are working on with "reduction"

    for (int reduction = 0; reduction < 4; reduction++)
    {
        fpreal pivot_value = 0;

        // This is the outer loop of the search for a pivot element
        // This loop finds the pivot element by choosing the element of the
        // matrix which has the largest absolute value of all elements of the
        // array not in columns/rows that contain a previously used pivot.
        for (int row = 0; row < 4; row++)
        {
            // if there hasn't already been a pivot in this row
            if ( !pivoted_columns_flags[row] )
            {
                // if there hasn't already been a pivot in column 0
                if ( !pivoted_columns_flags[0] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][0]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 0;
                    }
                }

                // if there hasn't already been a pivot in column 1
                if ( !pivoted_columns_flags[1] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][1]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 1;
                    }
                }

                // if there hasn't already been a pivot in column 2
                if ( !pivoted_columns_flags[2] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][2]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 2;
                    }
                }

                // if there hasn't already been a pivot in column 3
                if ( !pivoted_columns_flags[3] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][3]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 3;
                    }
                }
            }
        }

        // Odd check here if the matrix is filled with nan's or infinities
        // Verify that the pivot we found is not 0
        if (pivot_value <= tol)
        {
            return 1;
        }

        // Now we have found the pivot element for the current reduction.
        // This element of the matrix is the largest element of all elements
        // not in rows and columns previously used for other pivots.

        // Record that we have found a pivot in the current column
        pivoted_columns_flags[pivot_col] = true;
        indexrow[reduction] = pivot_row;
        indexcol[reduction] = pivot_col;

        // The pivot must be on the diagonal before we can use it. If the
        // pivot we found wasn't already on the diagonal, we swap rows to
        // put it there now.
        if ( pivot_row != pivot_col )
        {
            swapf(&(*matx)[pivot_row][0], &(*matx)[pivot_col][0]);
            swapf(&(*matx)[pivot_row][1], &(*matx)[pivot_col][1]);
            swapf(&(*matx)[pivot_row][2], &(*matx)[pivot_col][2]);
            swapf(&(*matx)[pivot_row][3], &(*matx)[pivot_col][3]);
        }


        // Note that from here on, the pivot is on the diagonal, and thus
        // has the same row index as column index. In particular, while
        // we may have swapped the row the pivot is in, we have not changed
        // the column. Thus, the pivot location will be referred to using
        // pivot_col only.

        switch (pivot_col)
        {
            case 0: mat4solvecol(matx, 0); break;
            case 1: mat4solvecol(matx, 1); break;
            case 2: mat4solvecol(matx, 2); break;
            case 3: mat4solvecol(matx, 3); break;
        }
    }


    // Finally, we "unscramble" the matrix, which was scrambled by
    // row swapping. This will produce the actual inverse of the matrix.
    for (int reduction = 3; reduction >= 0; reduction--)
    {
        int irow = indexrow[reduction];
        int icol = indexcol[reduction];
        if ( irow != icol )
        {
            swapf(&(*matx)[0][irow], &(*matx)[0][icol]);
            swapf(&(*matx)[1][irow], &(*matx)[1][icol]);
            swapf(&(*matx)[2][irow], &(*matx)[2][icol]);
            swapf(&(*matx)[3][irow], &(*matx)[3][icol]);
        }
    }

    if (is_trivial_col3)
    {
        // Force column 3 to be exact if input column 3 was exactly trivial.
        (*matx)[0][3] = 0;
        (*matx)[1][3] = 0;
        (*matx)[2][3] = 0;
        (*matx)[3][3] = 1;
    }

    return 0;
}


static fpreal 
mat2det(const mat2 m)
{
    return m.x * m.w - m.y * m.z;
}

static fpreal 
mat2inv(const mat2 m, mat2 *minvout)
{
    fpreal det = mat2det(m);
    if (det == 0)
        return 0;

    *minvout = (mat2)(
        m.w, -m.y,
        -m.z, m.x
    ) / det;
    return det;
}

// 3x3 matrix inversion that returns zero if |det(m)|<=tol.
static fpreal 
mat3invtol(const mat3 m, mat3 minvout, fpreal tol)
{
    fpreal det = det3(m);
    if (fabs(det) <= tol)
        return 0;

    // Inverse by cofactor method.
    minvout[0] = (fpreal3)(
         mat2det((mat2)(m[1].yz, m[2].yz)),
        -mat2det((mat2)(m[0].yz, m[2].yz)),
         mat2det((mat2)(m[0].yz, m[1].yz))
    ) / det;
    minvout[1] = (fpreal3)(
        -mat2det((mat2)(m[1].xz, m[2].xz)),
         mat2det((mat2)(m[0].xz, m[2].xz)),
        -mat2det((mat2)(m[0].xz, m[1].xz))        
    ) / det;
    minvout[2] = (fpreal3)(
         mat2det((mat2)(m[1].xy, m[2].xy)),
        -mat2det((mat2)(m[0].xy, m[2].xy)),
         mat2det((mat2)(m[0].xy, m[1].xy))
    ) / det;

    return det;
}

// 3x3 matrix inversion that returns zero if |det(m)|==0.
static fpreal 
mat3inv(const mat3 m, mat3 minvout)
{
    return mat3invtol(m, minvout, 0);
}

static void
mat3scale(mat3 mout, const mat3 m, fpreal scale)
{
    mout[0] = m[0] * scale;
    mout[1] = m[1] * scale;
    mout[2] = m[2] * scale;
}

// In-place version of above.
static 
void mat3scaleip(mat3 A, fpreal scale)
{
    A[0] *= scale;
    A[1] *= scale;
    A[2] *= scale;
}

static void
mat3lincomb2(mat3 mout, const mat3 m1, fpreal scale1, const mat3 m2, fpreal scale2)
{
    mout[0] = m1[0] * scale1 + m2[0] * scale2;
    mout[1] = m1[1] * scale1 + m2[1] * scale2;
    mout[2] = m1[2] * scale1 + m2[2] * scale2;
}

// Rotates the incoming positions with the given angle in degrees.
// It's more efficient to make a rotation matrix if you're going to rotate multiple matrices by the same angle
static fpreal2
rotate2D(fpreal2 pos, fpreal angle)
{
    angle = -angle * M_PI_F/180.0f;
    fpreal ca;
    fpreal sa = sincos(angle, &ca);
    mat2 rot = (mat2)(ca, sa, -sa, ca);
    return mat2vecmul(rot, pos);
}

// Check if matrix is diagonally dominant and has positive diagonal entries.
// This is a sufficient condition for positive-definiteness
// and can be used as an early out to avoid eigen decomposition.
static int
mat3posdiagdom(mat3 A)
{
    if (A[0].x < (fabs(A[0].y) + fabs(A[0].z)))
        return 0;
    if (A[1].y < (fabs(A[1].x) + fabs(A[1].z)))
        return 0;
    if (A[2].z < (fabs(A[2].x) + fabs(A[2].y)))
        return 0;
    return 1;
}

#ifndef NO_DOUBLE_SUPPORT
// Double versions of subset of the above functions
// (mainly for shapematching ATM).

typedef double3 mat3d[3];

static void
mat3fromcolsd(const double3 c0, const double3 c1, const double3 c2, mat3d m)
{
    m[0] = (double3)(c0.s0, c1.s0, c2.s0);
    m[1] = (double3)(c0.s1, c1.s1, c2.s1);
    m[2] = (double3)(c0.s2, c1.s2, c2.s2);
}

static void
transpose3d(const mat3d a, mat3d b)
{
    mat3fromcolsd(a[0], a[1], a[2], b);
}

#endif

#endif
#ifndef __RANDOM_H
#define __RANDOM_H

/******************************************************************************
 * HDK-consistent floor, integer hash, and fast random number generation.
 ******************************************************************************/

/// Returns the largest representable integer no greater than the given input.
static float
SYSfloorIL(float val)
{
    uint tmp = as_uint(val);
    uint shift = (tmp >> 23) & 0xff;

    if(shift < 0x7f)
    {
        return (tmp > 0x80000000) ? -1.0F : 0.0F;
    }
    else if(shift < 0x96)
    {
        uint mask = 0xffffffff << (0x96 - shift);
        if(tmp & 0x80000000)
        {
            if((tmp & ~mask) & 0x7fffff)
            {
                tmp &= mask;
                return as_float(tmp) - 1;
            }
            else
            {
                return val;
            }
        }
        else
        {
            return as_float(tmp & mask);
        }
    }
    else
    {
        return val;
    }
}

/// Consistent integer hash with the HDK.
static uint
SYSwang_inthash(uint key)
{
    key += ~(key << 16);
    key ^=  (key >>  5);
    key +=  (key <<  3);
    key ^=  (key >> 13);
    key += ~(key <<  9);
    key ^=  (key >> 17);
    return key;
}

/// Generates a uniform random number in the 0-1 range from the given seed and
/// updates the seed.
static float
SYSfastRandom(uint* seed)
{
    uint temp;
    *seed = (*seed) * 1664525 + 1013904223;
    temp = 0x3f800000 | (0x007fffff & (*seed));
    return as_float(temp) - 1.0f;
}

/******************************************************************************
 * Hashing macros for varying number of inputs.
 ******************************************************************************/

/// Conversion from a single float to an integer, for the clamped and unclamped
/// cases.
#define C_HASH1(x) ((int) SYSfloorIL(x))
#define U_HASH1(x) (as_uint(x))

/// Hash functions for 2-4 integers, used by clamped float hashing.
#define HASH2I(x, y) ((x^0xffffdead) * (y^0xffffc0de))
#define HASH3I(x, y, z) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169))
#define HASH4I(x, y, z, w) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169) \
                                           * (w^0xffff0397))
/// These macros declare hashing functions for clamped floating point numbers.
#define C_HASH2(x, y) HASH2I(C_HASH1(x), C_HASH1(y))
#define C_HASH3(x, y, z) HASH3I(C_HASH1(x), C_HASH1(y), C_HASH1(z))
#define C_HASH4(x, y, z, w) HASH4I(C_HASH1(x), C_HASH1(y), C_HASH1(z), \
                                   C_HASH1(w))

/// This macro hashes 2 integers, used by generic float hashing.
#define U_HASH2_RAW(x, y) y + SYSwang_inthash(x)
/// These macros declare hashing functions for generic floating point numbers.
#define U_HASH2(x, y) U_HASH2_RAW(U_HASH1(x), U_HASH1(y))
#define U_HASH3(x, y, z) U_HASH2_RAW(U_HASH2(x, y), U_HASH1(z))
#define U_HASH4(x, y, z, w) U_HASH2_RAW(U_HASH3(x, y, z), U_HASH1(w))

/******************************************************************************
 * Helper macros to hash inputs and generate outputs.
 ******************************************************************************/

/// Macros to hash the given number of floating point variables and store the
/// result in a new integer called hash.
#define C_1_HASH uint hash = SYSwang_inthash(C_HASH1(x));
#define C_2_HASH uint hash = SYSwang_inthash(C_HASH2(x, y));
#define C_3_HASH uint hash = SYSwang_inthash(C_HASH3(x, y, z));
#define C_4_HASH uint hash = SYSwang_inthash(C_HASH4(x, y, z, w));
#define U_1_HASH int hash = SYSwang_inthash(U_HASH1(x));
#define U_2_HASH int hash = SYSwang_inthash(U_HASH2(x, y));
#define U_3_HASH int hash = SYSwang_inthash(U_HASH3(x, y, z));
#define U_4_HASH int hash = SYSwang_inthash(U_HASH4(x, y, z, w));
/// Macros to return a vector of K random numbers; hash integer variable must be
/// declared and initialized.
#define RET_1_RAND return SYSfastRandom((uint*) &hash);
#define RET_2_RAND return (float2)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_3_RAND return (float3)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_4_RAND return (float4)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));

/******************************************************************************
 * Generation of final VEX_equivalent random_fhash() functions.
 ******************************************************************************/

static int VEXrandom_fhash_1(float x)
{
    U_1_HASH
    return hash;
}
static int VEXrandom_fhash_2(float x, float y)
{
    U_2_HASH
    return hash;
}
static int VEXrandom_fhash_3(float x, float y, float z)
{
    U_3_HASH
    return hash;
}
static int VEXrandom_fhash_4(float x, float y, float z, float w)
{
    U_4_HASH
    return hash;
}

/******************************************************************************
 * Generation of the final VEX-equivalent random() functions with float inputs.
 ******************************************************************************/

/// Macro that generates the code for random functions. NUM should be 2-4
/// (number of random numbers to return).
#define CREATE_RANDOM(NUM) \
static float ## NUM VEXrandom_1_ ## NUM(float x) \
{ \
    C_1_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_2_ ## NUM(float x, float y) \
{ \
    C_2_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_3_ ## NUM(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_4_ ## NUM(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point random.
#define CREATE_RANDOM_FLOAT \
static float VEXrandom_1_1(float x) \
{ \
    C_1_HASH \
    RET_1_RAND \
} \
static float VEXrandom_2_1(float x, float y) \
{ \
    C_2_HASH \
    RET_1_RAND \
} \
static float VEXrandom_3_1(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_1_RAND \
} \
static float VEXrandom_4_1(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RANDOM_FLOAT
CREATE_RANDOM(2)
CREATE_RANDOM(3)
CREATE_RANDOM(4)

/******************************************************************************
 * Generation of the final VEX-equivalent rand() functions.
 ******************************************************************************/

/// Macro that generates the code for rand functions. NUM should be 2-4 (number
/// of random numbers to return).
#define CREATE_RAND(NUM) \
static float ## NUM VEXrand_1_ ## NUM(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_2_ ## NUM(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_3_ ## NUM(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_4_ ## NUM(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point rand.
#define CREATE_RAND_FLOAT \
static float VEXrand_1_1(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_2_1(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_3_1(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_4_1(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RAND_FLOAT
CREATE_RAND(2)
CREATE_RAND(3)
CREATE_RAND(4)

#endif

#ifndef __IMX_H__
#define __IMX_H__

/******************************************************************************
 * VERBOSITY OPTIONS
 ******************************************************************************/
 
// By defining certain symbols before including this header, different error
// reporting options can be enabled. These options are listed in the table
// below.
//      SYMBOL                  ||          EFFECT
//  CHECK_RANGE                 ||  Report out-of-range indexing of buffers
//  CHECK_STORAGE_TYPE_READ     ||  Reports attempts to read floating point
//                              ||  values from integer layers or integer values
//                              ||  from floating point layers.
//  CHECK_STORAGE_TYPE_WRITE    ||  Reports attempts to write floating point
//                              ||  values to integer layers or integer values
//                              ||  to floating point layers.
//  CHECK_STORAGE_TYPE_CON      ||  Report if IMX_Layer storage type is not equal
//                              ||  to the (compile-time constant) storage argument
//  CHECK_CHANNEL_COUNT_CON     ||  Report if IMX_Layer channels is not equal
//                              ||  to the (compile-time constant) channels argument
// Note that the checks are done only when the respective symbol is defined, so
// these should only be used for validation and debugging purposes.

/******************************************************************************
 * STRUCTURES
 ******************************************************************************/

// Scale-translate transform.
typedef float4 STXform;

// Border type for a layer.
typedef enum
{
    IMX_CONSTANT,
    IMX_CLAMP,
    IMX_MIRROR,
    IMX_WRAP
} BorderType;

typedef enum
{
    IMX_TYPEINFO_NONE,
    IMX_TYPEINFO_COLOR,
    IMX_TYPEINFO_POSITION,
    IMX_TYPEINFO_VECTOR,
    IMX_TYPEINFO_NORMAL,
    IMX_TYPEINFO_OFFSETNORMAL,
    IMX_TYPEINFO_TEXTURE_COORD,
    IMX_TYPEINFO_ID,
    IMX_TYPEINFO_MASK,
    IMX_TYPEINFO_SDF,
    IMX_TYPEINFO_HEIGHT
} TypeInfoType;

// Type of data stored in a layer.
typedef enum
{
    INT8,
    INT16,
    INT32,
    FLOAT16,
    FLOAT32,
    FIXED8,
    FIXED16,
    // dummy values used to indicate IMX_Buffer::isConstant():
    CONSTANT_INT8,
    CONSTANT_INT16,
    CONSTANT_INT32,
    CONSTANT_FLOAT16,
    CONSTANT_FLOAT32,
    CONSTANT_FIXED8,
    CONSTANT_FIXED16
} StorageType;

// Type of projection.
typedef enum
{
    IMX_PROJ_ORTHOGRAPHIC,
    IMX_PROJ_PERSPECTIVE,
} ProjectionType;


// A structure containing metadata for a layer.
typedef struct
{
    float16                     image_to_world;
    float16                     world_to_image;
    float16                     camera_to_world;

    STXform                     buffer_to_image;
    STXform                     image_to_buffer;
    STXform                     buffer_to_pixel;
    float3                      camera_image_pos;

    float4                      default_f;
    int4                        default_i;

    int2                        resolution;
    int                         channels;
    int                         stride_x, stride_y;

    BorderType                  border;
    // Typeinfo is a hint and should only affect computation in the rarest
    // of situations.
    TypeInfoType                typeinfo;
    StorageType                 storage;
    ProjectionType              projection;
} IMX_Stat;

// A structure encapsulating a layer.
typedef struct
{
    global void* restrict       data;
    global IMX_Stat* restrict   stat;
} IMX_Layer;

/// multiply v by scale+translate xform
static float2 applySTXform(STXform, float2 v);
static float2 applySTXformInverse(STXform, float2 v);
static float2 applySTXformVec(STXform, float2 v);
static float2 applySTXformInverseVec(STXform, float2 v);

/// Space transfomrations;
static float2 bufferToImage(global const IMX_Stat* restrict, float2 xy);
static float2 imageToBuffer(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToPixel(global const IMX_Stat* restrict, float2 xy);
static float2 pixelToBuffer(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToTexture(global const IMX_Stat* restrict, float2 xy);
static float2 textureToBuffer(global const IMX_Stat* restrict, float2 xy);

static float3 imageToWorld(global const IMX_Stat* restrict, float2 xy);
static float3 image3ToWorld(global const IMX_Stat* restrict, float3 xy);
static float2 worldToImage(global const IMX_Stat* restrict, float3 xyz);
static float3 worldToImage3(global const IMX_Stat* restrict, float3 xyz);

/// Vector variants.
static float2 bufferToImageVec(global const IMX_Stat* restrict, float2 xy);
static float2 imageToBufferVec(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToPixelVec(global const IMX_Stat* restrict, float2 xy);
static float2 pixelToBufferVec(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToTextureVec(global const IMX_Stat* restrict, float2 xy);
static float2 textureToBufferVec(global const IMX_Stat* restrict, float2 xy);

static float3 imageToWorldVec(global const IMX_Stat* restrict, float2 xy);
static float3 image3ToWorldVec(global const IMX_Stat* restrict, float3 xy);
static float2 worldToImageVec(global const IMX_Stat* restrict, float3 xyz);
static float3 worldToImage3Vec(global const IMX_Stat* restrict, float3 xyz);

// The remaining functions are for implementing @ substitutions:
//
//  int @ix, @iy, @ixy                  // output buffer coordinate
//  int @xres                           // output buffer width
//  int @yres                           // output buffer height
//  int2 @res                           // output buffer (width,height)
//  int2 @tilesize                      // tile dimensions passed to CE_Snippet::execute()
// 
// Cur location: .image .pixel .texture suffixes specify space, image default
// @P supports .world as well.
//  float2 @P                           // image coordinate of output pixel
//  float2 @dPdx                        // derivative of @P per @ix
//  float2 @dPdy                        // derivative of @P per @iy
//  float2 @dPdxy                       // (@dPdx.x,@dPdy.y) rectangle for area
//                                      // sampling
//
// 'name' is replaced with the name of a layer binding:
//  void* @name.data                    // raw buffer data
//  IMX_Stat* @name.stat
//  int @name.xres                      // @name's buffer width
//  int @name.yres                      // @name's buffer height
//  float2 @name.res                    // @name's (width,height)
//  BorderType @name.border             // border type, compile-time constant
//  StorageType @name.storage           // data type, compile-time constant
//  int @name.channels                  // # of channels, compile-time constant
//  int @name.tuplesize                 // # of channels, compile-time constant
//
// Space transforms:
//  float2 @name.imageToBuffer(float2) 
//  float2 @name.bufferToImage(float2) 
//  float2 @name.pixelToBuffer(float2) 
//  float2 @name.bufferToPixel(float2) 
//  float2 @name.textureToBuffer(float2) 
//  float2 @name.bufferToTexture(float2) 
//  float3 @name.imageToWorld(float2)
//  float3 @name.image3ToWorld(float3)
//  float2 @name.worldToImage(float3)
//  float3 @name.worldToImage3(float3)
//
//  bool @name.bound                    // same as #ifdef HAS_name
//
//  T @name.bufferIndex(int2)           // value of buffer pixel, does tiling/borders
//  T @name.bufferSample(float2)        // bilinear interpolated (nearest for int)
//  T @name.imageNearest(float2)        // bufferIndex(rint(imageToBuffer(xy))
//  T @name.imageSample(float2)         // bufferSample(imageToBuffer(xy))
//  T @name.textureNearest(float2)      // bufferIndex(rint(textureToBuffer(xy))
//  T @name.textureSample(float2)       // bufferSample(textureToBuffer(xy))
//  T @name.worldSample(float3)         // bufferSample(imageToBuffer(worldToImage(xyz)))
//  T @name.worldNearest(float3)         // bufferIndex(rint(imageToBuffer(worldToImage(xyz))))
//  T @name                             // @name.imageSample(@P)
//  void @name.set(T v)                 // same as @name.setIndex((int2)(@ix,@iy), v)
//  void @name.setIndex(int2, T v)      // store value of buffer pixel, no test for out of range!
//
// Where T is not int:
//  T @name.dCdx(float2)                // derivative of @name.imageSample() per @ix
//  T @name.dCdx                        // @name.dCdx(@P)
//  T @name.dCdy(float2)                // derivative of @name.imageSample() per @iy
//  T @name.dCdy                        // @name.dCdy(@P)

// #include "imx_internal.h"
#endif

#ifndef __IMX_INTERNAL_H__
#define __IMX_INTERNAL_H__

// #include <matrix.h>

typedef float                   float1;
typedef int                     int1;

/// Converts image coordinates to linear index.
static int
_linearIndex(global const IMX_Stat* restrict stat, int2 xy)
{
#ifdef CHECK_RANGE
    if (xy.x < 0 || xy.x >= stat->resolution.x ||
        xy.y < 0 || xy.y >= stat->resolution.y)
        printf("Error: converting an invalid 2D index at %v2d to linear;"
               "resolution was %v2d.\n", xy, stat->resolution);
#endif
    return xy.x * stat->stride_x + xy.y * stat->stride_y;
}

/// Splits the given coordinates into integer and fractional parts.
static void
_splitCoordinates(float2 xy, int2* ix_p, float2* fx_p)
{
    float2 f;
    *fx_p = fract(xy, &f);
    *ix_p = convert_int2_sat(f);
}

/// Wraps the given coordinates for the specified resolution.
static int2
_wrapCoordinates(int2 xy, int2 res)
{
    int2 p = xy % res;
    return select(p, p + res, p < 0);
}

static int2
_mirrorCoordinates(int2 xy, int2 res)
{
    int2 res2 = res * 2;
    int2 p = _wrapCoordinates(xy, res2);
    return select(p, res2 - p - 1, p >= res);
}

/// Mirrors the given coordinates for the specified resolution. mirrored0 and
/// mirrored1 will have the mirrored versions of x and x+1, respectively.
static void
_mirrorCoordinates2(int2 xy, int2 res, int2* mirrored0, int2* mirrored1)
{
    int2 res2 = res * 2;
    int2 p = _wrapCoordinates(xy, res2);
    *mirrored0 = select(p, res2 - p - 1, p >= res);
    p = _wrapCoordinates(xy + 1, res2);
    *mirrored1 = select(p, res2 - p - 1, p >= res);
}

static bool
_outside(int2 xy, int2 res)
{
    return any(xy < 0) || any(xy >= res);
}

#ifdef CHECK_STORAGE_TYPE_CON
__constant char *
_getStorageName(StorageType storage)
{
    switch (storage)
    {
    case INT8:
        return "INT8";
    case INT16:
        return "INT16";
    case INT32:
        return "INT32";
    case FLOAT16:
        return "FLOAT16";
    case FLOAT32:
        return "FLOAT32";
    case FIXED8:
        return "FIXED8";
    case FIXED16:
        return "FIXED16";
    }
}
#define _CHECK_STORAGE(what) \
    if (storage != layer->stat->storage) \
        printf("Error: %s(%s), layer is %s\n", what, \
               _getStorageName(storage), _getStorageName(layer->stat->storage));
#else
#define _CHECK_STORAGE(what)
#endif

#ifdef CHECK_CHANNEL_COUNT_CON
#define _CHECK_CHANNEL(what) \
    if (channels != layer->stat->channels) \
        printf("Error: %s(channels=%d), layer is %d channels\n", \
               what, channels, layer->stat->channels);
#else
#define _CHECK_CHANNEL(what)
#endif

#ifdef CHECK_RANGE
#define _CHECK_RANGE(what) \
    if (index < 0 || index >= stat->resolution.y * stat->stride_y) \
        printf("Error: %s index %d out of range\n", what, index);
#else
#define _CHECK_RANGE(what)
#endif

#define CHECK_STAT(what) _CHECK_STORAGE(what) _CHECK_CHANNEL(what) _CHECK_RANGE(what)

#ifdef CHECK_STORAGE_TYPE_WRITE
#define WRITE_ERROR_I() printf("Error: writing integer values to a floating point layer.\n"); return
#define WRITE_ERROR_F() printf("Error: writing floating point values to an integer layer.\n"); return
#else
#define WRITE_ERROR_I() return
#define WRITE_ERROR_F() return
#endif

#ifdef CHECK_STORAGE_TYPE_READ
#define READ_ERROR_I() printf("Error: reading integer values from a floating point layer.\n"); return layer->stat->default_i
#define READ_ERROR_F() printf("Error: reading floating point values from an integer layer.\n"); return layer->stat->default_f
#else
#define READ_ERROR_I() return layer->stat->default_i
#define READ_ERROR_F() return layer->stat->default_f
#endif

// TODO: make sure rounding mode is good. (FIXED_POINT_STORAGE)
/// 8-bit fixed point conversion macros.
#define TO_FIXED8_SCALE 255
#define FROM_FIXED8_SCALE 0.00392156862f
#define FP_TO_FIXED8(v) convert_uchar_rte(clamp(v, 0.0f, 1.0f) \
                                          * TO_FIXED8_SCALE)
#define FP_TO_FIXED8_v(v, COMP) convert_uchar ## COMP ## _rte( \
    clamp(v, 0.0f, 1.0f) * TO_FIXED8_SCALE)
#define FIXED8_TO_FP(v) convert_float(v) * FROM_FIXED8_SCALE
#define FIXED8_TO_FP_v(v, COMP) convert_float ## COMP (v) \
    * FROM_FIXED8_SCALE
/// 16-bit fixed point conversion macros.
#define TO_FIXED16_SCALE 32767
#define FROM_FIXED16_SCALE 0.0000305185f
#define FP_TO_FIXED16(v) convert_short_rte(clamp(v, -1.0f, 1.0f) \
                                           * TO_FIXED16_SCALE)
#define FP_TO_FIXED16_v(v, COMP) convert_short ## COMP ## _rte( \
    clamp(v, -1.0f, 1.0f) * TO_FIXED16_SCALE)
#define FIXED16_TO_FP(v) convert_float(v) * FROM_FIXED16_SCALE
#define FIXED16_TO_FP_v(v, COMP) convert_float ## COMP (v) \
    * FROM_FIXED16_SCALE

static void
_setIndexLinI1a(IMX_Layer* layer, int index, int v, StorageType storage)
{
    switch (storage)
    {
    case INT8:
        ((global char*) layer->data)[index] = (char) v;
        return;
    case INT16:
        ((global short*) layer->data)[index] = (short) v;
        return;
    case INT32:
        ((global int*) layer->data)[index] = v;
        return;
    case FLOAT16:
        vstore_half_rte((float) v, index, (global half*)layer->data);
        break;
    case FLOAT32:
        ((global float*) layer->data)[index] = v;
        break;
    // TODO: should these clamp to 0 and 1 and convert to fixed point value?
    // (FIXED_POINT_STORAGE)
    case FIXED8:
        ((global uchar*) layer->data)[index] = (uchar) v;
        return;
    case FIXED16:
        ((global short*) layer->data)[index] = (short) v;
        return;
    default:
        WRITE_ERROR_I();
    }
}

static void
_setIndexLinF1a(IMX_Layer* layer, int index, float v, StorageType storage)
{
    switch (storage)
    {
    case FLOAT16:
        vstore_half_rte(v, index, (global half*) layer->data);
        break;
    case FLOAT32:
        ((global float*) layer->data)[index] = v;
        break;
    case FIXED8:
        ((global uchar*) layer->data)[index] = FP_TO_FIXED8(v);
        break;
    case FIXED16:
        ((global short*) layer->data)[index] = FP_TO_FIXED16(v);
        break;
    default:
        _setIndexLinI1a(layer, index, convert_int_sat_rtn(v+0.5f), storage);
    }
}

static void
_setIndexLinF2a(IMX_Layer* layer, int index, float2 v, StorageType storage)
{
    switch (storage)
    {
    case FLOAT16:
        vstore_half2_rte(v, index, (global half*) layer->data);
        break;
    case FLOAT32:
        vstore2(v, index, (global float*) layer->data);
        break;
    case FIXED8:
        vstore2(FP_TO_FIXED8_v(v, 2), index, (global uchar*) layer->data);
        break;
    case FIXED16:
        vstore2(FP_TO_FIXED16_v(v, 2), index, (global short*) layer->data);
        break;
    default:
        WRITE_ERROR_F();
    }
}

static void
_setIndexLinF3a(IMX_Layer* layer, int index, float3 v, StorageType storage)
{
    switch (storage)
    {
    case FLOAT16:
        vstore_half3_rte(v.xyz, index, (global half*) layer->data);
        break;
    case FLOAT32:
        vstore3(v.xyz, index, (global float*) layer->data);
        break;
    case FIXED8:
        vstore3(FP_TO_FIXED8_v(v, 3), index, (global uchar*) layer->data);
        break;
    case FIXED16:
        vstore3(FP_TO_FIXED16_v(v, 3), index, (global short*) layer->data);
        break;
    default:
        WRITE_ERROR_F();
    }
}

static void
_setIndexLinF4(IMX_Layer* layer, int index, float4 v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF4");
    switch (channels)
    {
    case 1:
        _setIndexLinF1a(layer, index, v.x, storage);
        break;
    case 2:
        _setIndexLinF2a(layer, index, v.xy, storage);
        break;
    case 3:
        _setIndexLinF3a(layer, index, v.xyz, storage);
        break;
    default:
        switch (storage)
        {
        case FLOAT16:
            vstore_half4_rte(v, index, (global half*) layer->data);
            break;
        case FLOAT32:
            vstore4(v, index, (global float*) layer->data);
            break;
        case FIXED8:
            vstore4(FP_TO_FIXED8_v(v, 4), index, (global uchar*) layer->data);
            break;
        case FIXED16:
            vstore4(FP_TO_FIXED16_v(v, 4), index, (global short*) layer->data);
            break;
        default:
            WRITE_ERROR_F();
        }
        break;
    }
}

static void
_setIndexLinI1(IMX_Layer* layer, int index, int v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexI1");
    if (channels == 1)
        _setIndexLinI1a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, v, storage, channels);
}

static void
_setIndexLinF1(IMX_Layer* layer, int index, float v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF1");
    if (channels == 1)
        _setIndexLinF1a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, v, storage, channels);
}

static void
_setIndexLinF2(IMX_Layer* layer, int index, float2 v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF2");
    if (channels == 2)
        _setIndexLinF2a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, (float4)(v,0.0f,1.0f), storage, channels);
}

static void
_setIndexLinF3(IMX_Layer* layer, int index, float3 v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF3");
    if (channels == 3)
        _setIndexLinF3a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, (float4)(v,1.0f), storage, channels);
}

////////////////////////////////////////////////////////////////////////////////

static int
_bufferIndexLinI1(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexI1");

    switch (storage)
    {
    case INT8:
        return ((global char*) layer->data)[index * channels];
    case INT16:
        return ((global short*) layer->data)[index * channels];
    case INT32:
        return ((global int*) layer->data)[index * channels];
    case CONSTANT_INT8:
        return *((global char*) layer->data);
    case CONSTANT_INT16:
        return *((global short*) layer->data);
    case CONSTANT_INT32:
        return *((global int*) layer->data);
    default:
        READ_ERROR_I().x;
    }
}

static float
_bufferIndexLinF1(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF1");

    switch (storage)
    {
    case FLOAT16:
        return vload_half(index*channels, (global half*) layer->data);
    case FLOAT32:
        return ((global float*) layer->data)[index*channels];
    case FIXED8:
        return FIXED8_TO_FP(((global uchar*) layer->data)[index*channels]);
    case FIXED16:
        return FIXED16_TO_FP(((global short*) layer->data)[index*channels]);
    case CONSTANT_FLOAT16:
        return vload_half(0, (global half*) layer->data);
    case CONSTANT_FLOAT32:
        return ((global float*) layer->data)[0];
    case CONSTANT_FIXED8:
        return FIXED8_TO_FP(((global uchar*) layer->data)[0]);
    case CONSTANT_FIXED16:
        return FIXED16_TO_FP(((global short*) layer->data)[0]);
    default:
        return _bufferIndexLinI1(layer, index, storage, channels);
    };
}

static float2
_bufferIndexLinF2(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF2");

    switch (channels)
    {
    case 1:
        return _bufferIndexLinF1(layer, index, storage, channels);
    default:
        switch (storage)
        {
        case FLOAT16:
            return vload_half2(0, (global half*) layer->data + index * channels);
        case FLOAT32:
            return vload2(0, (global float*) layer->data + index * channels);
        case FIXED8:
            return FIXED8_TO_FP_v(
                    vload2(0, (global uchar*) layer->data + index * channels),
                    2);
        case FIXED16:
            return FIXED16_TO_FP_v(
                    vload2(0, (global short*) layer->data + index * channels),
                    2);
        case CONSTANT_FLOAT16:
            return vload_half2(0, (global half*) layer->data);
        case CONSTANT_FLOAT32:
            return vload2(0, (global float*) layer->data);
        case CONSTANT_FIXED8:
            return FIXED8_TO_FP_v(vload2(0, (global uchar*) layer->data), 2);
        case CONSTANT_FIXED16:
            return FIXED16_TO_FP_v(vload2(0, (global short*) layer->data), 2);
        default:
            READ_ERROR_F().xy;
        }
    }
}

static float3
_bufferIndexLinF3(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF3");

    switch (channels)
    {
    case 1:
        return _bufferIndexLinF1(layer, index, storage, channels);
    case 2:
        return (float3)(_bufferIndexLinF2(layer, index, storage, channels), 0.0f);
    default:
        switch (storage)
        {
        case FLOAT16:
            return vload_half3(0, (global half*) layer->data + index * channels);
        case FLOAT32:
            return vload3(0, (global float*) layer->data + index * channels);
        case FIXED8:
            return FIXED8_TO_FP_v(
                    vload3(0, (global uchar*) layer->data + index * channels),
                    3);
        case FIXED16:
            return FIXED16_TO_FP_v(
                    vload3(0, (global short*) layer->data + index * channels),
                    3);
        case CONSTANT_FLOAT16:
            return vload_half3(0, (global half*) layer->data);
        case CONSTANT_FLOAT32:
            return vload3(0, (global float*) layer->data);
        case CONSTANT_FIXED8:
            return FIXED8_TO_FP_v(vload3(0, (global uchar*) layer->data), 3);
        case CONSTANT_FIXED16:
            return FIXED16_TO_FP_v(vload3(0, (global short*) layer->data), 3);
        default:
            READ_ERROR_F().xyz;
        }
    }
}

static float4
_bufferIndexLinF4(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF4");

    switch (channels)
    {
    case 1:
        return _bufferIndexLinF1(layer, index, storage, channels);
    case 2:
        return (float4)(_bufferIndexLinF2(layer, index, storage, channels), 0.0f, 1.0f);
    case 3:
        return (float4)(_bufferIndexLinF3(layer, index, storage, channels), 1.0f);
    default:
        switch (storage)
        {
        case FLOAT16:
            return vload_half4(index, (global half*) layer->data);
        case FLOAT32:
            return vload4(index, (global float*) layer->data);
        case FIXED8:
            return FIXED8_TO_FP_v(vload4(index, (global uchar*) layer->data),
                                  4);
        case FIXED16:
            return FIXED16_TO_FP_v(vload4(index, (global short*) layer->data),
                                   4);
        case CONSTANT_FLOAT16:
            return vload_half4(0, (global half*) layer->data);
        case CONSTANT_FLOAT32:
            return vload4(0, (global float*) layer->data);
        case CONSTANT_FIXED8:
            return FIXED8_TO_FP_v(vload4(0, (global uchar*) layer->data), 4);
        case CONSTANT_FIXED16:
            return FIXED16_TO_FP_v(vload4(0, (global short*) layer->data), 4);
        default:
            READ_ERROR_F();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#define _IMPLEMENT_I(DIM)                                            \
                                                                     \
static int ## DIM                                                    \
bufferIndexI ## DIM (const IMX_Layer* layer, int2 xy,                \
                     BorderType border, StorageType storage,         \
                     int channels)                                   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
    {                                                                \
        switch (border)                                              \
        {                                                            \
        case (IMX_CONSTANT):                                         \
            return 0.0f;                                             \
        case (IMX_CLAMP):                                            \
            xy = clamp(xy, (int2)(0), res - 1);                      \
            break;                                                   \
        case (IMX_MIRROR):                                           \
            xy = _mirrorCoordinates(xy, res);                        \
            break;                                                   \
        case (IMX_WRAP): default:                                    \
            xy = _wrapCoordinates(xy, res);                          \
            break;                                                   \
        }                                                            \
    }                                                                \
    return _bufferIndexLinI ## DIM (                                 \
        layer, _linearIndex(layer->stat, xy), storage, channels);    \
}                                                                    \
                                                                     \
static int ## DIM                                                    \
bufferSampleI ## DIM (const IMX_Layer* layer, float2 xy,             \
                      BorderType border, StorageType storage,        \
                      int channels)                                  \
{                                                                    \
    int2 c = convert_int2_sat_rtn(xy+0.5f);                          \
    return bufferIndexI ## DIM (layer, c, border, storage, channels);\
}                                                                    \
                                                                     \
static void                                                          \
_setIndexI ## DIM(IMX_Layer *layer, int2 xy, int ## DIM val, StorageType storage, int channels)   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
        return;                                                      \
    _setIndexLinI ## DIM(layer, _linearIndex(layer->stat, xy), val, storage, channels); \
}                                                                    \
/**/


_IMPLEMENT_I(1)
#undef _IMPLEMENT_I

#define _IMPLEMENT_F(DIM)                                            \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _CN(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 res = stat->resolution;                                     \
    float ## DIM k = 0.0f;                                           \
    float ## DIM v00 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    ix.x++;                                                          \
    float ## DIM v10 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    ix.y++;                                                          \
    float ## DIM v11 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    ix.x--;                                                          \
    float ## DIM v01 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _CL(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 res1 = stat->resolution - 1;                                \
    int2 ix0 = clamp(ix, (int2)(0), res1);                           \
    int2 ix1 = clamp(ix+1, (int2)(0), res1);                         \
    float ## DIM v00 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix0), storage, channels);          \
    float ## DIM v10 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix1.x, ix0.y)), storage, channels); \
    float ## DIM v01 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix0.x, ix1.y)), storage, channels); \
    float ## DIM v11 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix1), storage, channels);          \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _MR(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 ix0, ix1;                                                   \
    _mirrorCoordinates2(ix, stat->resolution, &ix0, &ix1);           \
    float ## DIM v00 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix0), storage, channels);          \
    float ## DIM v10 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix1.x, ix0.y)), storage, channels); \
    float ## DIM v01 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix0.x, ix1.y)), storage, channels); \
    float ## DIM v11 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix1), storage, channels);          \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _WR(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 ix0 = _wrapCoordinates(ix, stat->resolution);               \
    int2 ix1 = _wrapCoordinates(ix + 1, stat->resolution);           \
    float ## DIM v00 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix0), storage, channels);          \
    float ## DIM v10 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix1.x, ix0.y)), storage, channels); \
    float ## DIM v01 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix0.x, ix1.y)), storage, channels); \
    float ## DIM v11 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix1), storage, channels);          \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
bufferSampleF ## DIM (const IMX_Layer* layer, float2 xy,             \
                     BorderType border, StorageType storage,         \
                     int channels)                                   \
{                                                                    \
    switch (border)                                                  \
    {                                                                \
    case (IMX_CONSTANT):                                             \
        return _bufferSampleF ## DIM ## _CN(                         \
            layer, xy, storage, channels);                           \
    case (IMX_CLAMP):                                                \
        return _bufferSampleF ## DIM ## _CL(                         \
            layer, xy, storage, channels);                           \
    case (IMX_MIRROR):                                               \
        return _bufferSampleF ## DIM ## _MR(                         \
            layer, xy, storage, channels);                           \
    case (IMX_WRAP): default:                                        \
        return _bufferSampleF ## DIM ## _WR(                         \
            layer, xy, storage, channels);                           \
    }                                                                \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
bufferIndexF ## DIM (const IMX_Layer* layer, int2 xy,                \
                     BorderType border, StorageType storage,         \
                     int channels)                                   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
    {                                                                \
        switch (border)                                              \
        {                                                            \
        case (IMX_CONSTANT):                                         \
            return 0.0f;                                             \
        case (IMX_CLAMP):                                            \
            xy = clamp(xy, (int2)(0), res - 1);                      \
            break;                                                   \
        case (IMX_MIRROR):                                           \
            xy = _mirrorCoordinates(xy, res);                        \
            break;                                                   \
        case (IMX_WRAP): default:                                    \
            xy = _wrapCoordinates(xy, res);                          \
            break;                                                   \
        }                                                            \
    }                                                                \
    return _bufferIndexLinF ## DIM (                                 \
        layer, _linearIndex(layer->stat, xy), storage, channels);    \
}                                                                    \
                                                                     \
static void                                                          \
_setIndexF ## DIM(IMX_Layer *layer, int2 xy, float ## DIM val, StorageType storage, int channels)   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
        return;                                                      \
    _setIndexLinF ## DIM(layer, _linearIndex(layer->stat, xy), val, storage, channels); \
}                                                                    \
/**/

_IMPLEMENT_F(1)
_IMPLEMENT_F(2)
_IMPLEMENT_F(3)
_IMPLEMENT_F(4)
#undef _IMPLEMENT_F

static float2
applySTXform(STXform xform, float2 v)
{
    return v * xform.lo + xform.hi;
}

static float2
applySTXformInverse(STXform xform, float2 v)
{
    return (v - xform.hi) / xform.lo;
}

static float2
applySTXformVec(STXform xform, float2 v)
{
    return v * xform.lo;
}

static float2
applySTXformInverseVec(STXform xform, float2 v)
{
    return v / xform.lo;
}

static float2
bufferToImage(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXform(stat->buffer_to_image, v);
}

static float2
imageToBuffer(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXform(stat->image_to_buffer, v);
}

static float2
bufferToPixel(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXform(stat->buffer_to_pixel, v);
}

static float2
pixelToBuffer(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformInverse(stat->buffer_to_pixel, v);
}

static float2
bufferToTexture(global const IMX_Stat* restrict stat, float2 v)
{
    return (v + 0.5f) / (float2)(stat->resolution.x, stat->resolution.y);
}

static float2
textureToBuffer(global const IMX_Stat* restrict stat, float2 v)
{
    return v * ((float2)(stat->resolution.x, stat->resolution.y)) - 0.5f;
}

static float3
imageToWorld(global const IMX_Stat* restrict stat, float2 xy)
{
    float4      xyzw = 0;
    xyzw.xy = xy;
    xyzw.w = 1;
    xyzw = mat4vecmul(stat->image_to_world, xyzw);

    // No taper needed as we are on the plane.

    return xyzw.xyz;
}

static float3
image3ToWorld(global const IMX_Stat* restrict stat, float3 xyz)
{
    float4      xyzw = 0;
    xyzw.xyz = xyz;

    // We need to taper our image coordinates before
    // transforming back to world space.
    // Note z is not transformed here.
    if (stat->projection == IMX_PROJ_PERSPECTIVE)
    {
        float cameraz = stat->camera_image_pos.z;

        xyzw.xy *= (cameraz - xyzw.z) / cameraz;
    }

    xyzw.w = 1;
    xyzw = mat4vecmul(stat->image_to_world, xyzw);

    return xyzw.xyz;
}

static float2
worldToImage(global const IMX_Stat* restrict stat, float3 xyz)
{
    float4      xyzw = 0;
    xyzw.xyz = xyz;
    xyzw.w = 1;
    xyzw = mat4vecmul(stat->world_to_image, xyzw);

    // We are now in orthogonal coordinates that are "corrrect"
    // where the image plane is, so we need to account for the 
    // camera taper.
    if (stat->projection == IMX_PROJ_PERSPECTIVE)
    {
        float cameraz = stat->camera_image_pos.z;

        if (xyzw.z == cameraz)
            xyzw.xy = 0;
        else
            xyzw.xy *= cameraz / (cameraz - xyzw.z);
    }


    return xyzw.xy;
}

static float3
worldToImage3(global const IMX_Stat* restrict stat, float3 xyz)
{
    float4      xyzw = 0;
    xyzw.xyz = xyz;
    xyzw.w = 1;
    xyzw = mat4vecmul(stat->world_to_image, xyzw);

    // We are now in orthogonal coordinates that are "corrrect"
    // where the image plane is, so we need to account for the 
    // camera taper.
    if (stat->projection == IMX_PROJ_PERSPECTIVE)
    {
        float cameraz = stat->camera_image_pos.z;

        if (xyzw.z == cameraz)
            xyzw.xy = 0;
        else
            xyzw.xy *= cameraz / (cameraz - xyzw.z);
    }

    return xyzw.xyz;
}

static float2
bufferToImageVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformVec(stat->buffer_to_image, v);
}

static float2
imageToBufferVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformVec(stat->image_to_buffer, v);
}

static float2
bufferToPixelVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformVec(stat->buffer_to_pixel, v);
}

static float2
pixelToBufferVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformInverseVec(stat->buffer_to_pixel, v);
}

static float2
bufferToTextureVec(global const IMX_Stat* restrict stat, float2 v)
{
    return (v) / (float2)(stat->resolution.x, stat->resolution.y);
}

static float2
textureToBufferVec(global const IMX_Stat* restrict stat, float2 v)
{
    return v * ((float2)(stat->resolution.x, stat->resolution.y));
}

static float3
imageToWorldVec(global const IMX_Stat* restrict stat, float2 xy)
{
    float3      xyz = 0;
    xyz.xy = xy;
    xyz = mat43vec3mul(stat->image_to_world, xyz);

    // No taper needed as we are on the plane.

    return xyz;
}

static float3
image3ToWorldVec(global const IMX_Stat* restrict stat, float3 xyz)
{
    // Directions are not consistent in tapered spaces,
    // so we just punt and go with the isometric case.
    xyz = mat43vec3mul(stat->image_to_world, xyz);

    return xyz;
}

static float2
worldToImageVec(global const IMX_Stat* restrict stat, float3 xyz)
{
    xyz = mat43vec3mul(stat->world_to_image, xyz);

    // Directions are not consistent in tapered spaces,
    // so we just punt and go with the isometric case.
    return xyz.xy;
}

static float3
worldToImage3Vec(global const IMX_Stat* restrict stat, float3 xyz)
{
    xyz = mat43vec3mul(stat->world_to_image, xyz);

    // Directions are not consistent in tapered spaces,
    // so we just punt and go with the isometric case.
    return xyz;
}

static float4
dCdxF4(const IMX_Layer* layer, float2 ixy,
       BorderType border, StorageType storage, int channels,
       const IMX_Layer* dst)
{
    float2 xy = imageToBuffer(layer->stat, ixy);
    float d = (dst->stat->buffer_to_image.x * layer->stat->image_to_buffer.x) / 2;
    float4 a = bufferSampleF4(layer, (float2)(xy.x + d, xy.y), border, storage, channels);
    float4 b = bufferSampleF4(layer, (float2)(xy.x - d, xy.y), border, storage, channels);
    return a - b;
}

static float4
dCdxF4aligned(const IMX_Layer* layer, int2 xy,
              BorderType border, StorageType storage, int channels)
{
    float4 a = bufferIndexF4(layer, (int2)(xy.x+1, xy.y), border, storage, channels);
    float4 b = bufferIndexF4(layer, (int2)(xy.x-1, xy.y), border, storage, channels);
    return (a-b) * 0.5f;
}

static float4
dCdyF4(const IMX_Layer* layer, float2 ixy,
       BorderType border, StorageType storage, int channels,
       const IMX_Layer* dst)
{
    float2 xy = imageToBuffer(layer->stat, ixy);
    float d = (dst->stat->buffer_to_image.y * layer->stat->image_to_buffer.y) / 2;
    float4 a = bufferSampleF4(layer, (float2)(xy.x, xy.y + d), border, storage, channels);
    float4 b = bufferSampleF4(layer, (float2)(xy.x, xy.y - d), border, storage, channels);
    return a - b;
}

static float4
dCdyF4aligned(const IMX_Layer* layer, int2 xy,
              BorderType border, StorageType storage, int channels)
{
    float4 a = bufferIndexF4(layer, (int2)(xy.x, xy.y+1), border, storage, channels);
    float4 b = bufferIndexF4(layer, (int2)(xy.x, xy.y-1), border, storage, channels);
    return (a-b) * 0.5f;
}

#endif

#ifndef __IMX_FILTER_H__
#define __IMX_FILTER_H__

// Only a single filter can be used in a snippet. To choose filter:
//
// Either define one of the following:
//  FILTER_POINT
//  FILTER_BILINEAR
//  FILTER_BOX (floating-point width box, sometimes called a 'tent')
//  FILTER_TRIANGLE
//  FILTER_CUBIC (bicubic interpolation if scale==1)
//  FILTER_MITCHELL (non-interpolating cubic)
//  FILTER_BSPLINE (smooth non-interpolating cubic)
// Or define these symbols, this example is equivalent to FILTER_TRIANGLE:
//  #define FILTER_SIZE 2
//  static __constant float samples[FILTER_SIZE] = { 1, 0 };
//  #define FILTER samples
//  #define FILTER_SUPPORT 1
//
// The following @-substitutions can then be used to sample areas, for float1-4 bindings
// F @name.bufferSampleRect(float2 center, float2 size) -> area in buffer coordinates
// F @name.bufferSampleRectClip(float2 center, float2 size) -> area in buffer coordinates
// F @name.imageSampleRect(float2 center, float2 size) -> area in image coordinates
// F @name.imageSampleRectClip(float2 center, float2 size) -> area in image coordinates

// #include "imx_filter_internal.h"
#endif
#if (!defined(FILTER) || !defined(FILTER_SIZE) || !defined(FILTER_SUPPORT)) && \
    !defined(FILTER_POINT) && !defined(FILTER_BILINEAR) && \
    !defined(FILTER_BOX) && !defined(FILTER_TRIANGLE) && \
    !defined(FILTER_CUBIC) && !defined(FILTER_MITCHELL) && \
    !defined(FILTER_BSPLINE)
#define FILTER_POINT
#endif

#if defined(FILTER) && defined(FILTER_SIZE) && defined(FILTER_SUPPORT)
// this is so snippet can define it's own filter before including this

#elif defined(FILTER_POINT)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    // Make .5 round toward -infinity so a translate by .5 does not throw away every other pixel
    return bufferIndexF4(layer, convert_int2_sat_rtn(xy + 0.5f), border, storage, channels);
}

#elif defined(FILTER_BILINEAR)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    return bufferSampleF4(layer, xy, border, storage, channels);
}

#elif defined(FILTER_BOX)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    xy = clamp(xy, -1e8f, 1e8f);
    dxy = clamp(dxy, 1.0f, 67.0f);
    float2 r = (dxy + 1.0f) / 2;
    int2 ia; float2 fa;
    _splitCoordinates(r - xy, &ia, &fa);
    ia = -ia;
    int2 ib; float2 fb;
    _splitCoordinates(xy + r, &ib, &fb);
    float4 sum = 0;
    int2 ixy;
    float w = fa.y;
    for (ixy.y = ia.y; ixy.y <= ib.y;)
    {
        ixy.x = ia.x;
        float4 sum1 = fa.x * bufferIndexF4(layer, ixy, border, storage, channels);
        for (ixy.x++; ixy.x < ib.x; ixy.x++)
            sum1 += bufferIndexF4(layer, ixy, border, storage, channels);
        sum1 += fb.x * bufferIndexF4(layer, ixy, border, storage, channels);
        sum += w * sum1;
        w = (++ixy.y) < ib.y ? 1.0f : fb.y;
    }
    return sum / (dxy.x * dxy.y);
}

#elif defined(FILTER_TRIANGLE)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    float2 r = clamp(dxy, 1.0f, 1.0e8f);
    int2 ia = convert_int2_sat_rtp(xy - r);
    int2 ib = convert_int2_sat_rtp(xy + r);
    float4 sum = 0;
    float div = 0;
    int2 ixy;
    int2 inc = (ib - ia) / 61 + 1;
    for (ixy.y = ia.y; ixy.y < ib.y; ixy.y += inc.y)
    {
        float4 sum1 = 0;
        float div1 = 0;
        for (ixy.x = ia.x; ixy.x < ib.x; ixy.x += inc.x)
        {
            float w = 1 - fabs(ixy.x - xy.x) / r.x;
            sum1 += w * bufferIndexF4(layer, ixy, border, storage, channels);
            div1 += w;
        }
        float w = 1 - fabs(ixy.y - xy.y) / r.y;
        sum += w * sum1;
        div += w * div1;
    }
    return sum / div;
}

#elif defined(FILTER_CUBIC)
#define FILTER_SIZE 25
static __constant float Cubic[FILTER_SIZE] = { // mitchell(25, 0, 0.5)
    6, 5.9010415, 5.625, 5.203125, 4.6666665, 4.046875, 3.375, 2.6822917,
    2, 1.359375, 0.7916667, 0.328125, 0, -0.21006945, -0.3472222, -0.421875,
    -0.44444445, -0.4253472, -0.375, -0.30381945, -0.22222222, -0.140625, -0.06944445, -0.019097222,
    0};
#define FILTER Cubic
#define FILTER_SUPPORT 2

#elif defined(FILTER_MITCHELL)
#define FILTER_SIZE 25
static __constant float Mitchell[FILTER_SIZE] = { // mitchell(25, 1/3.0, 1/3.0)
    5.3333335, 5.2540507, 5.0324073, 4.6927085, 4.259259, 3.7563658, 3.2083333, 2.6394675,
    2.074074, 1.5364584, 1.050926, 0.6417824, 0.33333334, 0.116705246, -0.038580246, -0.140625,
    -0.19753087, -0.21739969, -0.20833333, -0.17843364, -0.13580246, -0.088541664, -0.044753086, -0.01253858,
    0}; //-1.7763568e-15};
#define FILTER Mitchell
#define FILTER_SUPPORT 2

#elif defined(FILTER_BSPLINE)
#define FILTER_SIZE 25
static __constant float BSpline[FILTER_SIZE] = { // mitchell(25, 1, 0)
    4, 3.9600694, 3.8472223, 3.671875, 3.4444444, 3.1753473, 2.875, 2.5538194,
    2.2222223, 1.890625, 1.5694444, 1.2690972, 1, 0.7702546, 0.5787037, 0.421875,
    0.2962963, 0.19849537, 0.125, 0.07233796, 0.037037037, 0.015625, 0.0046296297, 0.0005787037,
    0};
#define FILTER BSpline
#define FILTER_SUPPORT 2

#endif

#if defined(FILTER)

static float
sampleLookup(constant float * in, float pos)
{
    float flr;
    float t = fract(pos, &flr);
    int flooridx = convert_int(flr);
    int ceilidx = flooridx+1;
    ceilidx = min(ceilidx, FILTER_SIZE-1);
    return mix(in[flooridx], in[ceilidx], t);
}

static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    float2 r = min(max(dxy, 1.0f) * FILTER_SUPPORT, 1.0e8f);
    int2 ia = convert_int2_sat_rtp(xy - r);
    int2 ib = convert_int2_sat_rtp(xy + r);
    float2 k = (FILTER_SIZE - 1) / r;
    float4 sum = 0;
    float div = 0;
    int2 ixy;
    int2 inc = (ib - ia) / 61 + 1;
    for (ixy.y = ia.y; ixy.y < ib.y; ixy.y += inc.y)
    {
        float4 sum1 = 0;
        float div1 = 0;
        for (ixy.x = ia.x; ixy.x < ib.x; ixy.x += inc.x)
        {
            float w = sampleLookup(FILTER, fabs(ixy.x - xy.x) * k.x);
            sum1 += w * bufferIndexF4(layer, ixy, border, storage, channels);
            div1 += w;
        }
        float w = sampleLookup(FILTER, fabs(ixy.y - xy.y) * k.y);
        sum += w * sum1;
        div += w * div1;
    }
    return sum / div;
}
#endif

// Intersects the incoming sampling window with the layer's footprint and
// returns the filtered value in that rectangle. (Thus, this should never read
// outside pixels.)
static float4
bufferSampleRectClipF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                       StorageType storage, int channels)
{
#if defined(FILTER_POINT)
    return bufferSampleRectF4(layer, xy, dxy, IMX_CONSTANT, storage, channels);
#else
    float2 wh = convert_float2(layer->stat->resolution);
    dxy *= 0.5f;
    float2 xy0 = max(xy - dxy, -0.5f);
    float2 xy1 = min(xy + dxy, wh - 0.5f);
    if (any(xy0 >= xy1))
        return 0.0f;
    return bufferSampleRectF4(layer, (xy1 + xy0) * 0.5f, xy1 - xy0, IMX_CLAMP,
                              storage, channels);
#endif
}

static float4
constImageSampleRectClip(float2 xy, float2 dxy, float4 defval)
{
    // Clip to image space; if there is any intersection, return defval;
    // otherwise, return 0 since we're fully outside the image.
    float2 xy0 = max(xy - dxy, -1.0f);
    float2 xy1 = min(xy + dxy, 1.0f);
    return any(xy0 >= xy1) ? 0.0f : defval;
}

// convert derivatives (or a parallelogram) to nearest ortho rectangle
static float2
wh_from_dP(float2 dPdx, float2 dPdy)
{
    return hypot(dPdx, dPdy);
}

/* Python to generate the filters
def mitchell(n, B, C):
    print("[%d] = {"%n, end='');
    for i in range(0,n):
        if i: print(",", end='')
        if not(i%8): print("\n   ", end='')
        x = (2.0*i)/(n-1)
        if (x < 1):
            v = ((12-9*B-6*C)*x-18+12*B+6*C)*x*x+6-2*B;
        else:
            v = (((-B-6*C)*x+6*B+30*C)*x-12*B-48*C)*x+8*B+24*C;
        print(" %.6g"%v, end='')
    print("};")

mitchell(25, 1/3.0, 1/3.0)
*/
#define AT_elemnum      _bound_idx
#define AT_ix   _bound_gidx
#define AT_iy   _bound_gidy
#define AT_ixy  (int2)(_bound_gidx, _bound_gidy)
#define AT_res  (_RUNOVER_LAYER.stat->resolution)
#define AT_xres (_RUNOVER_LAYER.stat->resolution.x)
#define AT_yres (_RUNOVER_LAYER.stat->resolution.y)
#define AT_P_image      _bound_P_image
#define AT_P_pixel      _bound_P_pixel
#define AT_P_texture    _bound_P_texture
#define AT_P_world      (imageToWorld(_RUNOVER_LAYER.stat, _bound_P_image))
#define AT_P    AT_P_image
#define AT_dPdx_image   ((float2)(_RUNOVER_LAYER.stat->buffer_to_image.x,0))
#define AT_dPdx_pixel   ((float2)(_RUNOVER_LAYER.stat->buffer_to_pixel.x,0))
#define AT_dPdx_texture ((float2)(1.0f/(float)_RUNOVER_LAYER.stat->resolution.x,0))
#define AT_dPdx AT_dPdx_image
#define AT_dPdy_image   ((float2)(0, _RUNOVER_LAYER.stat->buffer_to_image.y))
#define AT_dPdy_pixel   ((float2)(0, _RUNOVER_LAYER.stat->buffer_to_pixel.y))
#define AT_dPdy_texture ((float2)(0, 1.0f/(float)_RUNOVER_LAYER.stat->resolution.y))
#define AT_dPdy AT_dPdy_image
#define AT_dPdxy_image (_RUNOVER_LAYER.stat->buffer_to_image.xy)
#define AT_dPdxy_pixel (_RUNOVER_LAYER.stat->buffer_to_pixel.xy)
#define AT_dPdxy_texture ((float2)(1.0f/(float)_RUNOVER_LAYER.stat->resolution.x,1.0f/(float)_RUNOVER_LAYER.stat->resolution.y))
#define AT_dPdxy AT_dPdxy_image
#define AT_tilesize     _bound_tilesize
#define AT_amp  _bound_amp
#define AT_center       _bound_center
#define AT_featuresize  _bound_featuresize
#define AT_off  _bound_off
#define AT_roughness    _bound_roughness
#define AT_octaves      _bound_octaves
#define AT_lacunarity   _bound_lacunarity
#define AT_tiledsize    _bound_tiledsize
#define AT_istiled      _bound_istiled
#define AT_noisetype    _bound_noisetype
#define AT_usetime      _bound_usetime
#define AT_timeoffset   _bound_timeoffset
#define AT_timescale    _bound_timescale
#define AT_parmtime     _bound_parmtime
#define AT_is3d _bound_is3d
#define AT_pulselength  _bound_pulselength
#define AT_doloop       _bound_doloop
#define AT_looptime     _bound_looptime
#define AT_percomp      _bound_percomp
#define AT_contrastexponent     _bound_contrastexponent
#define AT_post_dofold  _bound_post_dofold
#define AT_post_docomplement    _bound_post_docomplement
#define AT_post_dobias  _bound_post_dobias
#define AT_post_bias    _bound_post_bias
#define AT_post_dogain  _bound_post_dogain
#define AT_post_gain    _bound_post_gain
#define AT_post_dogamma _bound_post_dogamma
#define AT_post_gamma   _bound_post_gamma
#define AT_post_docontrast      _bound_post_docontrast
#define AT_post_contrast        _bound_post_contrast
#define AT_post_doclampmin      _bound_post_doclampmin
#define AT_post_minimum _bound_post_minimum
#define AT_post_doclampmax      _bound_post_doclampmax
#define AT_post_maximum _bound_post_maximum
#define AT_jitter       _bound_jitter
#define AT_noise_data   _bound_noise
#define AT_noise_bound  1
#define AT_noise_stat   ((global IMX_Stat * restrict) _bound_noise_stat_void)
#define AT_noise_layer  &_bound_noise_layer
#define AT_noise_border _bound_noise_border
#define AT_noise_storage        _bound_noise_storage
#define AT_noise_channels       _bound_noise_channels
#define AT_noise_tuplesize      _bound_noise_channels
#define AT_noise_xres   _bound_noise_layer.stat->resolution.x
#define AT_noise_yres   _bound_noise_layer.stat->resolution.y
#define AT_noise_res    convert_float2(_bound_noise_layer.stat->resolution)
#define AT_noise_set(_val_)     _setIndexLinF4(&_bound_noise_layer, _bound_idx, _val_, _bound_noise_storage, _bound_noise_channels)
#define AT_noise_setIndex(_xy_, _val_)  _setIndexF4(&_bound_noise_layer, _xy_, _val_, _bound_noise_storage, _bound_noise_channels)
#define AT_noise_bufferToImage(_xy_)    (bufferToImage(AT_noise_stat, _xy_))
#define AT_noise_imageToBuffer(_xy_)    (imageToBuffer(AT_noise_stat, _xy_))
#define AT_noise_bufferToPixel(_xy_)    (bufferToPixel(AT_noise_stat, _xy_))
#define AT_noise_pixelToBuffer(_xy_)    (pixelToBuffer(AT_noise_stat, _xy_))
#define AT_noise_bufferToTexture(_xy_)  (bufferToTexture(AT_noise_stat, _xy_))
#define AT_noise_textureToBuffer(_xy_)  (textureToBuffer(AT_noise_stat, _xy_))
#define AT_noise_imageToWorld(_xy_)     (imageToWorld(AT_noise_stat, _xy_))
#define AT_noise_image3ToWorld(_xyz_)   (image3ToWorld(AT_noise_stat, _xyz_))
#define AT_noise_worldToImage(_xyz_)    (worldToImage(AT_noise_stat, _xyz_))
#define AT_noise_worldToImage3(_xyz_)   (worldToImage3(AT_noise_stat, _xyz_))
#ifdef HAS_src
#define AT_src_data     _bound_src
#else
#define AT_src_data     0
#endif
#ifdef HAS_src
#define AT_src_bound    1
#else
#define AT_src_bound    0
#endif
#ifdef HAS_src
#define AT_src_stat     ((global IMX_Stat * restrict) _bound_src_stat_void)
#else
#define AT_src_stat     0
#endif
#ifdef HAS_src
#define AT_src_layer    &_bound_src_layer
#else
#define AT_src_layer    0
#endif
#ifdef HAS_src
#define AT_src_border   _bound_src_border
#else
#define AT_src_border   IMX_WRAP
#endif
#ifdef HAS_src
#define AT_src_storage  _bound_src_storage
#else
#define AT_src_storage  FLOAT32
#endif
#ifdef HAS_src
#define AT_src_channels _bound_src_channels
#else
#define AT_src_channels 4
#endif
#ifdef HAS_src
#define AT_src_tuplesize        _bound_src_channels
#else
#define AT_src_tuplesize        4
#endif
#ifdef HAS_src
#define AT_src_xres     _bound_src_layer.stat->resolution.x
#else
#define AT_src_xres     1
#endif
#ifdef HAS_src
#define AT_src_yres     _bound_src_layer.stat->resolution.y
#else
#define AT_src_yres     1
#endif
#ifdef HAS_src
#define AT_src_res      convert_float2(_bound_src_layer.stat->resolution)
#else
#define AT_src_res      (float2)(1)
#endif
#ifdef HAS_src
#define AT_src_bufferToImage(_xy_)      (bufferToImage(AT_src_stat, _xy_))
#else
#define AT_src_bufferToImage(_xy_)      (_xy_)
#endif
#ifdef HAS_src
#define AT_src_imageToBuffer(_xy_)      (imageToBuffer(AT_src_stat, _xy_))
#else
#define AT_src_imageToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_src
#define AT_src_bufferToPixel(_xy_)      (bufferToPixel(AT_src_stat, _xy_))
#else
#define AT_src_bufferToPixel(_xy_)      (_xy_)
#endif
#ifdef HAS_src
#define AT_src_pixelToBuffer(_xy_)      (pixelToBuffer(AT_src_stat, _xy_))
#else
#define AT_src_pixelToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_src
#define AT_src_bufferToTexture(_xy_)    (bufferToTexture(AT_src_stat, _xy_))
#else
#define AT_src_bufferToTexture(_xy_)    (_xy_)
#endif
#ifdef HAS_src
#define AT_src_textureToBuffer(_xy_)    (textureToBuffer(AT_src_stat, _xy_))
#else
#define AT_src_textureToBuffer(_xy_)    (_xy_)
#endif
#ifdef HAS_src
#define AT_src_imageToWorld(_xy_)       (imageToWorld(AT_src_stat, _xy_))
#else
#define AT_src_imageToWorld(_xy_)       ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_src
#define AT_src_image3ToWorld(_xyz_)     (image3ToWorld(AT_src_stat, _xyz_))
#else
#define AT_src_image3ToWorld(_xyz_)     (_xyz_)
#endif
#ifdef HAS_src
#define AT_src_worldToImage(_xyz_)      (worldToImage(AT_src_stat, _xyz_))
#else
#define AT_src_worldToImage(_xyz_)      ((_xyz_).xy)
#endif
#ifdef HAS_src
#define AT_src_worldToImage3(_xyz_)     (worldToImage3(AT_src_stat, _xyz_))
#else
#define AT_src_worldToImage3(_xyz_)     (_xyz_)
#endif
#ifdef HAS_pos
#define AT_pos_data     _bound_pos
#else
#define AT_pos_data     0
#endif
#ifdef HAS_pos
#define AT_pos_bound    1
#else
#define AT_pos_bound    0
#endif
#ifdef HAS_pos
#define AT_pos_stat     ((global IMX_Stat * restrict) _bound_pos_stat_void)
#else
#define AT_pos_stat     0
#endif
#ifdef HAS_pos
#define AT_pos_layer    &_bound_pos_layer
#else
#define AT_pos_layer    0
#endif
#ifdef HAS_pos
#define AT_pos_border   _bound_pos_border
#else
#define AT_pos_border   IMX_WRAP
#endif
#ifdef HAS_pos
#define AT_pos_storage  _bound_pos_storage
#else
#define AT_pos_storage  FLOAT32
#endif
#ifdef HAS_pos
#define AT_pos_channels _bound_pos_channels
#else
#define AT_pos_channels 4
#endif
#ifdef HAS_pos
#define AT_pos_tuplesize        _bound_pos_channels
#else
#define AT_pos_tuplesize        4
#endif
#ifdef HAS_pos
#define AT_pos_xres     _bound_pos_layer.stat->resolution.x
#else
#define AT_pos_xres     1
#endif
#ifdef HAS_pos
#define AT_pos_yres     _bound_pos_layer.stat->resolution.y
#else
#define AT_pos_yres     1
#endif
#ifdef HAS_pos
#define AT_pos_res      convert_float2(_bound_pos_layer.stat->resolution)
#else
#define AT_pos_res      (float2)(1)
#endif
#define CONSTANT1(s) CONSTANT_ ## s
#define CONSTANT_(s) CONSTANT1(s)
#ifdef CONSTANT_pos
#define pos_args2 CONSTANT_(_bound_pos_storage), _bound_pos_channels
#else
#define pos_args2 _bound_pos_storage, _bound_pos_channels
#endif
#define pos_args3 _bound_pos_border, pos_args2
#ifdef HAS_pos
#define AT_pos_bufferIndex(_xy_)        bufferIndexF2(&_bound_pos_layer, _xy_, pos_args3)
#else
#define AT_pos_bufferIndex(_xy_)        _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_bufferSample(_xy_)       bufferSampleF2(&_bound_pos_layer, _xy_, pos_args3)
#else
#define AT_pos_bufferSample(_xy_)       _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_imageNearest(_xy_)       bufferIndexF2(&_bound_pos_layer, convert_int2_sat_rtn(imageToBuffer(AT_pos_stat, _xy_) + 0.5f), pos_args3)
#else
#define AT_pos_imageNearest(_xy_)       _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_imageSample(_xy_)        bufferSampleF2(&_bound_pos_layer, imageToBuffer(AT_pos_stat, _xy_), pos_args3)
#else
#define AT_pos_imageSample(_xy_)        _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_worldNearest(_xyz_)      bufferIndexF2(&_bound_pos_layer, convert_int2_sat_rtn(imageToBuffer(AT_pos_stat, worldToImage(AT_pos_stat, _xyz_)) + 0.5f), pos_args3)
#else
#define AT_pos_worldNearest(_xyz_)      _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_worldSample(_xyz_)       bufferSampleF2(&_bound_pos_layer, imageToBuffer(AT_pos_stat, worldToImage(AT_pos_stat, _xyz_)), pos_args3)
#else
#define AT_pos_worldSample(_xyz_)       _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_textureNearest(_xy_)     bufferIndexF2(&_bound_pos_layer, convert_int2_sat_rtn(textureToBuffer(AT_pos_stat, _xy_) + 0.5f), pos_args3)
#else
#define AT_pos_textureNearest(_xy_)     _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_textureSample(_xy_)      bufferSampleF2(&_bound_pos_layer, textureToBuffer(AT_pos_stat, _xy_), pos_args3)
#else
#define AT_pos_textureSample(_xy_)      _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_1(_xy_)  bufferSampleF2(&_bound_pos_layer, imageToBuffer(AT_pos_stat, _xy_), pos_args3)
#else
#define AT_pos_1(_xy_)  _bound_pos
#endif
#ifdef HAS_pos
#ifdef ALIGNED_pos
#define AT_pos  _bufferIndexLinF2(&_bound_pos_layer, _bound_idx, pos_args2)
#else
#define AT_pos  bufferSampleF2(&_bound_pos_layer, imageToBuffer(AT_pos_stat, _bound_P_image), pos_args3)
#endif
#else
#define AT_pos  _bound_pos
#endif
#ifdef HAS_pos
#ifdef ALIGNED_pos
#define AT_pos_dCdx     dCdxF4aligned(&_bound_pos_layer, (int2)(_bound_gidx, _bound_gidy), pos_args3).xy
#else
#define AT_pos_dCdx     dCdxF4(&_bound_pos_layer, _bound_P_image, pos_args3, &_RUNOVER_LAYER).xy
#endif
#else
#define AT_pos_dCdx     ((float2)0)
#endif
#ifdef HAS_pos
#define AT_pos_dCdx_1(_xy_)     dCdxF4(&_bound_pos_layer, _xy_, pos_args3, &_RUNOVER_LAYER).xy
#else
#define AT_pos_dCdx_1(_xy_)     ((float2)0)
#endif
#ifdef HAS_pos
#ifdef ALIGNED_pos
#define AT_pos_dCdy     dCdyF4aligned(&_bound_pos_layer, (int2)(_bound_gidx, _bound_gidy), pos_args3).xy
#else
#define AT_pos_dCdy     dCdyF4(&_bound_pos_layer, _bound_P_image, pos_args3, &_RUNOVER_LAYER).xy
#endif
#else
#define AT_pos_dCdy     ((float2)0)
#endif
#ifdef HAS_pos
#define AT_pos_dCdy_1(_xy_)     dCdyF4(&_bound_pos_layer, _xy_, pos_args3, &_RUNOVER_LAYER).xy
#else
#define AT_pos_dCdy_1(_xy_)     ((float2)0)
#endif
#ifdef HAS_pos
#define AT_pos_bufferSampleRect(_xy_, _dxy_)    bufferSampleRectF4(&_bound_pos_layer, _xy_, _dxy_, pos_args3).xy
#else
#define AT_pos_bufferSampleRect(_xy_, _dxy_)    _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_bufferSampleRectClip(_xy_, _dxy_)        bufferSampleRectClipF4(&_bound_pos_layer, _xy_, _dxy_, pos_args2).xy
#else
#define AT_pos_bufferSampleRectClip(_xy_, _dxy_)        constImageSampleRectClip(bufferToImage(AT_pos_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_pos_stat->resolution.x, AT_pos_stat->resolution.y)), _bound_pos).xy
#endif
#ifdef HAS_pos
#define AT_pos_imageSampleRect(_xy_, _dxy_)     AT_pos_bufferSampleRect(imageToBuffer(AT_pos_stat, _xy_), AT_pos_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_pos_imageSampleRect(_xy_, _dxy_)     _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_imageSampleRectClip(_xy_, _dxy_) AT_pos_bufferSampleRectClip(imageToBuffer(AT_pos_stat, _xy_), AT_pos_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_pos_imageSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(_xy_, _dxy_, _bound_pos).xy
#endif
#ifdef HAS_pos
#define AT_pos_textureSampleRect(_xy_, _dxy_)   AT_pos_bufferSampleRect(textureToBuffer(AT_pos_stat, _xy_), (float2)(AT_pos_stat->resolution.x, AT_pos_stat->resolution.y) * (_dxy_))
#else
#define AT_pos_textureSampleRect(_xy_, _dxy_)   _bound_pos
#endif
#ifdef HAS_pos
#define AT_pos_textureSampleRectClip(_xy_, _dxy_)       AT_pos_bufferSampleRectClip(textureToBuffer(AT_pos_stat, _xy_), (float2)(AT_pos_stat->resolution.x, AT_pos_stat->resolution.y) * (_dxy_))
#else
#define AT_pos_textureSampleRectClip(_xy_, _dxy_)       constImageSampleRectClip(bufferToImage(AT_pos_stat, textureToBuffer(AT_pos_stat, _xy_)), _dxy_ * ((float2)(AT_pos_stat->resolution.x, AT_pos_stat->resolution.y)) * AT_pos_stat->buffer_to_image.lo, _bound_pos).xy
#endif
#ifdef HAS_pos
#define AT_pos_bufferToImage(_xy_)      (bufferToImage(AT_pos_stat, _xy_))
#else
#define AT_pos_bufferToImage(_xy_)      (_xy_)
#endif
#ifdef HAS_pos
#define AT_pos_imageToBuffer(_xy_)      (imageToBuffer(AT_pos_stat, _xy_))
#else
#define AT_pos_imageToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_pos
#define AT_pos_bufferToPixel(_xy_)      (bufferToPixel(AT_pos_stat, _xy_))
#else
#define AT_pos_bufferToPixel(_xy_)      (_xy_)
#endif
#ifdef HAS_pos
#define AT_pos_pixelToBuffer(_xy_)      (pixelToBuffer(AT_pos_stat, _xy_))
#else
#define AT_pos_pixelToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_pos
#define AT_pos_bufferToTexture(_xy_)    (bufferToTexture(AT_pos_stat, _xy_))
#else
#define AT_pos_bufferToTexture(_xy_)    (_xy_)
#endif
#ifdef HAS_pos
#define AT_pos_textureToBuffer(_xy_)    (textureToBuffer(AT_pos_stat, _xy_))
#else
#define AT_pos_textureToBuffer(_xy_)    (_xy_)
#endif
#ifdef HAS_pos
#define AT_pos_imageToWorld(_xy_)       (imageToWorld(AT_pos_stat, _xy_))
#else
#define AT_pos_imageToWorld(_xy_)       ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_pos
#define AT_pos_image3ToWorld(_xyz_)     (image3ToWorld(AT_pos_stat, _xyz_))
#else
#define AT_pos_image3ToWorld(_xyz_)     (_xyz_)
#endif
#ifdef HAS_pos
#define AT_pos_worldToImage(_xyz_)      (worldToImage(AT_pos_stat, _xyz_))
#else
#define AT_pos_worldToImage(_xyz_)      ((_xyz_).xy)
#endif
#ifdef HAS_pos
#define AT_pos_worldToImage3(_xyz_)     (worldToImage3(AT_pos_stat, _xyz_))
#else
#define AT_pos_worldToImage3(_xyz_)     (_xyz_)
#endif
#ifdef HAS_intime
#define AT_intime_data  _bound_intime
#else
#define AT_intime_data  0
#endif
#ifdef HAS_intime
#define AT_intime_bound 1
#else
#define AT_intime_bound 0
#endif
#ifdef HAS_intime
#define AT_intime_stat  ((global IMX_Stat * restrict) _bound_intime_stat_void)
#else
#define AT_intime_stat  0
#endif
#ifdef HAS_intime
#define AT_intime_layer &_bound_intime_layer
#else
#define AT_intime_layer 0
#endif
#ifdef HAS_intime
#define AT_intime_border        _bound_intime_border
#else
#define AT_intime_border        IMX_WRAP
#endif
#ifdef HAS_intime
#define AT_intime_storage       _bound_intime_storage
#else
#define AT_intime_storage       FLOAT32
#endif
#ifdef HAS_intime
#define AT_intime_channels      _bound_intime_channels
#else
#define AT_intime_channels      4
#endif
#ifdef HAS_intime
#define AT_intime_tuplesize     _bound_intime_channels
#else
#define AT_intime_tuplesize     4
#endif
#ifdef HAS_intime
#define AT_intime_xres  _bound_intime_layer.stat->resolution.x
#else
#define AT_intime_xres  1
#endif
#ifdef HAS_intime
#define AT_intime_yres  _bound_intime_layer.stat->resolution.y
#else
#define AT_intime_yres  1
#endif
#ifdef HAS_intime
#define AT_intime_res   convert_float2(_bound_intime_layer.stat->resolution)
#else
#define AT_intime_res   (float2)(1)
#endif
#ifdef CONSTANT_intime
#define intime_args2 CONSTANT_(_bound_intime_storage), _bound_intime_channels
#else
#define intime_args2 _bound_intime_storage, _bound_intime_channels
#endif
#define intime_args3 _bound_intime_border, intime_args2
#ifdef HAS_intime
#define AT_intime_bufferIndex(_xy_)     bufferIndexF1(&_bound_intime_layer, _xy_, intime_args3)
#else
#define AT_intime_bufferIndex(_xy_)     _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_bufferSample(_xy_)    bufferSampleF1(&_bound_intime_layer, _xy_, intime_args3)
#else
#define AT_intime_bufferSample(_xy_)    _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_imageNearest(_xy_)    bufferIndexF1(&_bound_intime_layer, convert_int2_sat_rtn(imageToBuffer(AT_intime_stat, _xy_) + 0.5f), intime_args3)
#else
#define AT_intime_imageNearest(_xy_)    _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_imageSample(_xy_)     bufferSampleF1(&_bound_intime_layer, imageToBuffer(AT_intime_stat, _xy_), intime_args3)
#else
#define AT_intime_imageSample(_xy_)     _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_worldNearest(_xyz_)   bufferIndexF1(&_bound_intime_layer, convert_int2_sat_rtn(imageToBuffer(AT_intime_stat, worldToImage(AT_intime_stat, _xyz_)) + 0.5f), intime_args3)
#else
#define AT_intime_worldNearest(_xyz_)   _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_worldSample(_xyz_)    bufferSampleF1(&_bound_intime_layer, imageToBuffer(AT_intime_stat, worldToImage(AT_intime_stat, _xyz_)), intime_args3)
#else
#define AT_intime_worldSample(_xyz_)    _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_textureNearest(_xy_)  bufferIndexF1(&_bound_intime_layer, convert_int2_sat_rtn(textureToBuffer(AT_intime_stat, _xy_) + 0.5f), intime_args3)
#else
#define AT_intime_textureNearest(_xy_)  _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_textureSample(_xy_)   bufferSampleF1(&_bound_intime_layer, textureToBuffer(AT_intime_stat, _xy_), intime_args3)
#else
#define AT_intime_textureSample(_xy_)   _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_1(_xy_)       bufferSampleF1(&_bound_intime_layer, imageToBuffer(AT_intime_stat, _xy_), intime_args3)
#else
#define AT_intime_1(_xy_)       _bound_intime
#endif
#ifdef HAS_intime
#ifdef ALIGNED_intime
#define AT_intime       _bufferIndexLinF1(&_bound_intime_layer, _bound_idx, intime_args2)
#else
#define AT_intime       bufferSampleF1(&_bound_intime_layer, imageToBuffer(AT_intime_stat, _bound_P_image), intime_args3)
#endif
#else
#define AT_intime       _bound_intime
#endif
#ifdef HAS_intime
#ifdef ALIGNED_intime
#define AT_intime_dCdx  dCdxF4aligned(&_bound_intime_layer, (int2)(_bound_gidx, _bound_gidy), intime_args3).x
#else
#define AT_intime_dCdx  dCdxF4(&_bound_intime_layer, _bound_P_image, intime_args3, &_RUNOVER_LAYER).x
#endif
#else
#define AT_intime_dCdx  0.0f
#endif
#ifdef HAS_intime
#define AT_intime_dCdx_1(_xy_)  dCdxF4(&_bound_intime_layer, _xy_, intime_args3, &_RUNOVER_LAYER).x
#else
#define AT_intime_dCdx_1(_xy_)  0.0f
#endif
#ifdef HAS_intime
#ifdef ALIGNED_intime
#define AT_intime_dCdy  dCdyF4aligned(&_bound_intime_layer, (int2)(_bound_gidx, _bound_gidy), intime_args3).x
#else
#define AT_intime_dCdy  dCdyF4(&_bound_intime_layer, _bound_P_image, intime_args3, &_RUNOVER_LAYER).x
#endif
#else
#define AT_intime_dCdy  0.0f
#endif
#ifdef HAS_intime
#define AT_intime_dCdy_1(_xy_)  dCdyF4(&_bound_intime_layer, _xy_, intime_args3, &_RUNOVER_LAYER).x
#else
#define AT_intime_dCdy_1(_xy_)  0.0f
#endif
#ifdef HAS_intime
#define AT_intime_bufferSampleRect(_xy_, _dxy_) bufferSampleRectF4(&_bound_intime_layer, _xy_, _dxy_, intime_args3).x
#else
#define AT_intime_bufferSampleRect(_xy_, _dxy_) _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_bufferSampleRectClip(_xy_, _dxy_)     bufferSampleRectClipF4(&_bound_intime_layer, _xy_, _dxy_, intime_args2).x
#else
#define AT_intime_bufferSampleRectClip(_xy_, _dxy_)     constImageSampleRectClip(bufferToImage(AT_intime_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_intime_stat->resolution.x, AT_intime_stat->resolution.y)), _bound_intime).x
#endif
#ifdef HAS_intime
#define AT_intime_imageSampleRect(_xy_, _dxy_)  AT_intime_bufferSampleRect(imageToBuffer(AT_intime_stat, _xy_), AT_intime_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_intime_imageSampleRect(_xy_, _dxy_)  _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_imageSampleRectClip(_xy_, _dxy_)      AT_intime_bufferSampleRectClip(imageToBuffer(AT_intime_stat, _xy_), AT_intime_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_intime_imageSampleRectClip(_xy_, _dxy_)      constImageSampleRectClip(_xy_, _dxy_, _bound_intime).x
#endif
#ifdef HAS_intime
#define AT_intime_textureSampleRect(_xy_, _dxy_)        AT_intime_bufferSampleRect(textureToBuffer(AT_intime_stat, _xy_), (float2)(AT_intime_stat->resolution.x, AT_intime_stat->resolution.y) * (_dxy_))
#else
#define AT_intime_textureSampleRect(_xy_, _dxy_)        _bound_intime
#endif
#ifdef HAS_intime
#define AT_intime_textureSampleRectClip(_xy_, _dxy_)    AT_intime_bufferSampleRectClip(textureToBuffer(AT_intime_stat, _xy_), (float2)(AT_intime_stat->resolution.x, AT_intime_stat->resolution.y) * (_dxy_))
#else
#define AT_intime_textureSampleRectClip(_xy_, _dxy_)    constImageSampleRectClip(bufferToImage(AT_intime_stat, textureToBuffer(AT_intime_stat, _xy_)), _dxy_ * ((float2)(AT_intime_stat->resolution.x, AT_intime_stat->resolution.y)) * AT_intime_stat->buffer_to_image.lo, _bound_intime).x
#endif
#ifdef HAS_intime
#define AT_intime_bufferToImage(_xy_)   (bufferToImage(AT_intime_stat, _xy_))
#else
#define AT_intime_bufferToImage(_xy_)   (_xy_)
#endif
#ifdef HAS_intime
#define AT_intime_imageToBuffer(_xy_)   (imageToBuffer(AT_intime_stat, _xy_))
#else
#define AT_intime_imageToBuffer(_xy_)   (_xy_)
#endif
#ifdef HAS_intime
#define AT_intime_bufferToPixel(_xy_)   (bufferToPixel(AT_intime_stat, _xy_))
#else
#define AT_intime_bufferToPixel(_xy_)   (_xy_)
#endif
#ifdef HAS_intime
#define AT_intime_pixelToBuffer(_xy_)   (pixelToBuffer(AT_intime_stat, _xy_))
#else
#define AT_intime_pixelToBuffer(_xy_)   (_xy_)
#endif
#ifdef HAS_intime
#define AT_intime_bufferToTexture(_xy_) (bufferToTexture(AT_intime_stat, _xy_))
#else
#define AT_intime_bufferToTexture(_xy_) (_xy_)
#endif
#ifdef HAS_intime
#define AT_intime_textureToBuffer(_xy_) (textureToBuffer(AT_intime_stat, _xy_))
#else
#define AT_intime_textureToBuffer(_xy_) (_xy_)
#endif
#ifdef HAS_intime
#define AT_intime_imageToWorld(_xy_)    (imageToWorld(AT_intime_stat, _xy_))
#else
#define AT_intime_imageToWorld(_xy_)    ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_intime
#define AT_intime_image3ToWorld(_xyz_)  (image3ToWorld(AT_intime_stat, _xyz_))
#else
#define AT_intime_image3ToWorld(_xyz_)  (_xyz_)
#endif
#ifdef HAS_intime
#define AT_intime_worldToImage(_xyz_)   (worldToImage(AT_intime_stat, _xyz_))
#else
#define AT_intime_worldToImage(_xyz_)   ((_xyz_).xy)
#endif
#ifdef HAS_intime
#define AT_intime_worldToImage3(_xyz_)  (worldToImage3(AT_intime_stat, _xyz_))
#else
#define AT_intime_worldToImage3(_xyz_)  (_xyz_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_data       _bound_inoctaves
#else
#define AT_inoctaves_data       0
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_bound      1
#else
#define AT_inoctaves_bound      0
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_stat       ((global IMX_Stat * restrict) _bound_inoctaves_stat_void)
#else
#define AT_inoctaves_stat       0
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_layer      &_bound_inoctaves_layer
#else
#define AT_inoctaves_layer      0
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_border     _bound_inoctaves_border
#else
#define AT_inoctaves_border     IMX_WRAP
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_storage    _bound_inoctaves_storage
#else
#define AT_inoctaves_storage    FLOAT32
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_channels   _bound_inoctaves_channels
#else
#define AT_inoctaves_channels   4
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_tuplesize  _bound_inoctaves_channels
#else
#define AT_inoctaves_tuplesize  4
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_xres       _bound_inoctaves_layer.stat->resolution.x
#else
#define AT_inoctaves_xres       1
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_yres       _bound_inoctaves_layer.stat->resolution.y
#else
#define AT_inoctaves_yres       1
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_res        convert_float2(_bound_inoctaves_layer.stat->resolution)
#else
#define AT_inoctaves_res        (float2)(1)
#endif
#ifdef CONSTANT_inoctaves
#define inoctaves_args2 CONSTANT_(_bound_inoctaves_storage), _bound_inoctaves_channels
#else
#define inoctaves_args2 _bound_inoctaves_storage, _bound_inoctaves_channels
#endif
#define inoctaves_args3 _bound_inoctaves_border, inoctaves_args2
#ifdef HAS_inoctaves
#define AT_inoctaves_bufferIndex(_xy_)  bufferIndexF1(&_bound_inoctaves_layer, _xy_, inoctaves_args3)
#else
#define AT_inoctaves_bufferIndex(_xy_)  _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_bufferSample(_xy_) bufferSampleF1(&_bound_inoctaves_layer, _xy_, inoctaves_args3)
#else
#define AT_inoctaves_bufferSample(_xy_) _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_imageNearest(_xy_) bufferIndexF1(&_bound_inoctaves_layer, convert_int2_sat_rtn(imageToBuffer(AT_inoctaves_stat, _xy_) + 0.5f), inoctaves_args3)
#else
#define AT_inoctaves_imageNearest(_xy_) _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_imageSample(_xy_)  bufferSampleF1(&_bound_inoctaves_layer, imageToBuffer(AT_inoctaves_stat, _xy_), inoctaves_args3)
#else
#define AT_inoctaves_imageSample(_xy_)  _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_worldNearest(_xyz_)        bufferIndexF1(&_bound_inoctaves_layer, convert_int2_sat_rtn(imageToBuffer(AT_inoctaves_stat, worldToImage(AT_inoctaves_stat, _xyz_)) + 0.5f), inoctaves_args3)
#else
#define AT_inoctaves_worldNearest(_xyz_)        _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_worldSample(_xyz_) bufferSampleF1(&_bound_inoctaves_layer, imageToBuffer(AT_inoctaves_stat, worldToImage(AT_inoctaves_stat, _xyz_)), inoctaves_args3)
#else
#define AT_inoctaves_worldSample(_xyz_) _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_textureNearest(_xy_)       bufferIndexF1(&_bound_inoctaves_layer, convert_int2_sat_rtn(textureToBuffer(AT_inoctaves_stat, _xy_) + 0.5f), inoctaves_args3)
#else
#define AT_inoctaves_textureNearest(_xy_)       _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_textureSample(_xy_)        bufferSampleF1(&_bound_inoctaves_layer, textureToBuffer(AT_inoctaves_stat, _xy_), inoctaves_args3)
#else
#define AT_inoctaves_textureSample(_xy_)        _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_1(_xy_)    bufferSampleF1(&_bound_inoctaves_layer, imageToBuffer(AT_inoctaves_stat, _xy_), inoctaves_args3)
#else
#define AT_inoctaves_1(_xy_)    _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#ifdef ALIGNED_inoctaves
#define AT_inoctaves    _bufferIndexLinF1(&_bound_inoctaves_layer, _bound_idx, inoctaves_args2)
#else
#define AT_inoctaves    bufferSampleF1(&_bound_inoctaves_layer, imageToBuffer(AT_inoctaves_stat, _bound_P_image), inoctaves_args3)
#endif
#else
#define AT_inoctaves    _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#ifdef ALIGNED_inoctaves
#define AT_inoctaves_dCdx       dCdxF4aligned(&_bound_inoctaves_layer, (int2)(_bound_gidx, _bound_gidy), inoctaves_args3).x
#else
#define AT_inoctaves_dCdx       dCdxF4(&_bound_inoctaves_layer, _bound_P_image, inoctaves_args3, &_RUNOVER_LAYER).x
#endif
#else
#define AT_inoctaves_dCdx       0.0f
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_dCdx_1(_xy_)       dCdxF4(&_bound_inoctaves_layer, _xy_, inoctaves_args3, &_RUNOVER_LAYER).x
#else
#define AT_inoctaves_dCdx_1(_xy_)       0.0f
#endif
#ifdef HAS_inoctaves
#ifdef ALIGNED_inoctaves
#define AT_inoctaves_dCdy       dCdyF4aligned(&_bound_inoctaves_layer, (int2)(_bound_gidx, _bound_gidy), inoctaves_args3).x
#else
#define AT_inoctaves_dCdy       dCdyF4(&_bound_inoctaves_layer, _bound_P_image, inoctaves_args3, &_RUNOVER_LAYER).x
#endif
#else
#define AT_inoctaves_dCdy       0.0f
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_dCdy_1(_xy_)       dCdyF4(&_bound_inoctaves_layer, _xy_, inoctaves_args3, &_RUNOVER_LAYER).x
#else
#define AT_inoctaves_dCdy_1(_xy_)       0.0f
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_inoctaves_layer, _xy_, _dxy_, inoctaves_args3).x
#else
#define AT_inoctaves_bufferSampleRect(_xy_, _dxy_)      _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_inoctaves_layer, _xy_, _dxy_, inoctaves_args2).x
#else
#define AT_inoctaves_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_inoctaves_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_inoctaves_stat->resolution.x, AT_inoctaves_stat->resolution.y)), _bound_inoctaves).x
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_imageSampleRect(_xy_, _dxy_)       AT_inoctaves_bufferSampleRect(imageToBuffer(AT_inoctaves_stat, _xy_), AT_inoctaves_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_inoctaves_imageSampleRect(_xy_, _dxy_)       _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_imageSampleRectClip(_xy_, _dxy_)   AT_inoctaves_bufferSampleRectClip(imageToBuffer(AT_inoctaves_stat, _xy_), AT_inoctaves_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_inoctaves_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_inoctaves).x
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_textureSampleRect(_xy_, _dxy_)     AT_inoctaves_bufferSampleRect(textureToBuffer(AT_inoctaves_stat, _xy_), (float2)(AT_inoctaves_stat->resolution.x, AT_inoctaves_stat->resolution.y) * (_dxy_))
#else
#define AT_inoctaves_textureSampleRect(_xy_, _dxy_)     _bound_inoctaves
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_textureSampleRectClip(_xy_, _dxy_) AT_inoctaves_bufferSampleRectClip(textureToBuffer(AT_inoctaves_stat, _xy_), (float2)(AT_inoctaves_stat->resolution.x, AT_inoctaves_stat->resolution.y) * (_dxy_))
#else
#define AT_inoctaves_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_inoctaves_stat, textureToBuffer(AT_inoctaves_stat, _xy_)), _dxy_ * ((float2)(AT_inoctaves_stat->resolution.x, AT_inoctaves_stat->resolution.y)) * AT_inoctaves_stat->buffer_to_image.lo, _bound_inoctaves).x
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_bufferToImage(_xy_)        (bufferToImage(AT_inoctaves_stat, _xy_))
#else
#define AT_inoctaves_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_imageToBuffer(_xy_)        (imageToBuffer(AT_inoctaves_stat, _xy_))
#else
#define AT_inoctaves_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_bufferToPixel(_xy_)        (bufferToPixel(AT_inoctaves_stat, _xy_))
#else
#define AT_inoctaves_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_pixelToBuffer(_xy_)        (pixelToBuffer(AT_inoctaves_stat, _xy_))
#else
#define AT_inoctaves_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_bufferToTexture(_xy_)      (bufferToTexture(AT_inoctaves_stat, _xy_))
#else
#define AT_inoctaves_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_textureToBuffer(_xy_)      (textureToBuffer(AT_inoctaves_stat, _xy_))
#else
#define AT_inoctaves_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_imageToWorld(_xy_) (imageToWorld(AT_inoctaves_stat, _xy_))
#else
#define AT_inoctaves_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_image3ToWorld(_xyz_)       (image3ToWorld(AT_inoctaves_stat, _xyz_))
#else
#define AT_inoctaves_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_worldToImage(_xyz_)        (worldToImage(AT_inoctaves_stat, _xyz_))
#else
#define AT_inoctaves_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_inoctaves
#define AT_inoctaves_worldToImage3(_xyz_)       (worldToImage3(AT_inoctaves_stat, _xyz_))
#else
#define AT_inoctaves_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_data     _bound_inroughness
#else
#define AT_inroughness_data     0
#endif
#ifdef HAS_inroughness
#define AT_inroughness_bound    1
#else
#define AT_inroughness_bound    0
#endif
#ifdef HAS_inroughness
#define AT_inroughness_stat     ((global IMX_Stat * restrict) _bound_inroughness_stat_void)
#else
#define AT_inroughness_stat     0
#endif
#ifdef HAS_inroughness
#define AT_inroughness_layer    &_bound_inroughness_layer
#else
#define AT_inroughness_layer    0
#endif
#ifdef HAS_inroughness
#define AT_inroughness_border   _bound_inroughness_border
#else
#define AT_inroughness_border   IMX_WRAP
#endif
#ifdef HAS_inroughness
#define AT_inroughness_storage  _bound_inroughness_storage
#else
#define AT_inroughness_storage  FLOAT32
#endif
#ifdef HAS_inroughness
#define AT_inroughness_channels _bound_inroughness_channels
#else
#define AT_inroughness_channels 4
#endif
#ifdef HAS_inroughness
#define AT_inroughness_tuplesize        _bound_inroughness_channels
#else
#define AT_inroughness_tuplesize        4
#endif
#ifdef HAS_inroughness
#define AT_inroughness_xres     _bound_inroughness_layer.stat->resolution.x
#else
#define AT_inroughness_xres     1
#endif
#ifdef HAS_inroughness
#define AT_inroughness_yres     _bound_inroughness_layer.stat->resolution.y
#else
#define AT_inroughness_yres     1
#endif
#ifdef HAS_inroughness
#define AT_inroughness_res      convert_float2(_bound_inroughness_layer.stat->resolution)
#else
#define AT_inroughness_res      (float2)(1)
#endif
#ifdef CONSTANT_inroughness
#define inroughness_args2 CONSTANT_(_bound_inroughness_storage), _bound_inroughness_channels
#else
#define inroughness_args2 _bound_inroughness_storage, _bound_inroughness_channels
#endif
#define inroughness_args3 _bound_inroughness_border, inroughness_args2
#ifdef HAS_inroughness
#define AT_inroughness_bufferIndex(_xy_)        bufferIndexF1(&_bound_inroughness_layer, _xy_, inroughness_args3)
#else
#define AT_inroughness_bufferIndex(_xy_)        _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_bufferSample(_xy_)       bufferSampleF1(&_bound_inroughness_layer, _xy_, inroughness_args3)
#else
#define AT_inroughness_bufferSample(_xy_)       _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_imageNearest(_xy_)       bufferIndexF1(&_bound_inroughness_layer, convert_int2_sat_rtn(imageToBuffer(AT_inroughness_stat, _xy_) + 0.5f), inroughness_args3)
#else
#define AT_inroughness_imageNearest(_xy_)       _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_imageSample(_xy_)        bufferSampleF1(&_bound_inroughness_layer, imageToBuffer(AT_inroughness_stat, _xy_), inroughness_args3)
#else
#define AT_inroughness_imageSample(_xy_)        _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_worldNearest(_xyz_)      bufferIndexF1(&_bound_inroughness_layer, convert_int2_sat_rtn(imageToBuffer(AT_inroughness_stat, worldToImage(AT_inroughness_stat, _xyz_)) + 0.5f), inroughness_args3)
#else
#define AT_inroughness_worldNearest(_xyz_)      _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_worldSample(_xyz_)       bufferSampleF1(&_bound_inroughness_layer, imageToBuffer(AT_inroughness_stat, worldToImage(AT_inroughness_stat, _xyz_)), inroughness_args3)
#else
#define AT_inroughness_worldSample(_xyz_)       _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_textureNearest(_xy_)     bufferIndexF1(&_bound_inroughness_layer, convert_int2_sat_rtn(textureToBuffer(AT_inroughness_stat, _xy_) + 0.5f), inroughness_args3)
#else
#define AT_inroughness_textureNearest(_xy_)     _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_textureSample(_xy_)      bufferSampleF1(&_bound_inroughness_layer, textureToBuffer(AT_inroughness_stat, _xy_), inroughness_args3)
#else
#define AT_inroughness_textureSample(_xy_)      _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_1(_xy_)  bufferSampleF1(&_bound_inroughness_layer, imageToBuffer(AT_inroughness_stat, _xy_), inroughness_args3)
#else
#define AT_inroughness_1(_xy_)  _bound_inroughness
#endif
#ifdef HAS_inroughness
#ifdef ALIGNED_inroughness
#define AT_inroughness  _bufferIndexLinF1(&_bound_inroughness_layer, _bound_idx, inroughness_args2)
#else
#define AT_inroughness  bufferSampleF1(&_bound_inroughness_layer, imageToBuffer(AT_inroughness_stat, _bound_P_image), inroughness_args3)
#endif
#else
#define AT_inroughness  _bound_inroughness
#endif
#ifdef HAS_inroughness
#ifdef ALIGNED_inroughness
#define AT_inroughness_dCdx     dCdxF4aligned(&_bound_inroughness_layer, (int2)(_bound_gidx, _bound_gidy), inroughness_args3).x
#else
#define AT_inroughness_dCdx     dCdxF4(&_bound_inroughness_layer, _bound_P_image, inroughness_args3, &_RUNOVER_LAYER).x
#endif
#else
#define AT_inroughness_dCdx     0.0f
#endif
#ifdef HAS_inroughness
#define AT_inroughness_dCdx_1(_xy_)     dCdxF4(&_bound_inroughness_layer, _xy_, inroughness_args3, &_RUNOVER_LAYER).x
#else
#define AT_inroughness_dCdx_1(_xy_)     0.0f
#endif
#ifdef HAS_inroughness
#ifdef ALIGNED_inroughness
#define AT_inroughness_dCdy     dCdyF4aligned(&_bound_inroughness_layer, (int2)(_bound_gidx, _bound_gidy), inroughness_args3).x
#else
#define AT_inroughness_dCdy     dCdyF4(&_bound_inroughness_layer, _bound_P_image, inroughness_args3, &_RUNOVER_LAYER).x
#endif
#else
#define AT_inroughness_dCdy     0.0f
#endif
#ifdef HAS_inroughness
#define AT_inroughness_dCdy_1(_xy_)     dCdyF4(&_bound_inroughness_layer, _xy_, inroughness_args3, &_RUNOVER_LAYER).x
#else
#define AT_inroughness_dCdy_1(_xy_)     0.0f
#endif
#ifdef HAS_inroughness
#define AT_inroughness_bufferSampleRect(_xy_, _dxy_)    bufferSampleRectF4(&_bound_inroughness_layer, _xy_, _dxy_, inroughness_args3).x
#else
#define AT_inroughness_bufferSampleRect(_xy_, _dxy_)    _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_bufferSampleRectClip(_xy_, _dxy_)        bufferSampleRectClipF4(&_bound_inroughness_layer, _xy_, _dxy_, inroughness_args2).x
#else
#define AT_inroughness_bufferSampleRectClip(_xy_, _dxy_)        constImageSampleRectClip(bufferToImage(AT_inroughness_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_inroughness_stat->resolution.x, AT_inroughness_stat->resolution.y)), _bound_inroughness).x
#endif
#ifdef HAS_inroughness
#define AT_inroughness_imageSampleRect(_xy_, _dxy_)     AT_inroughness_bufferSampleRect(imageToBuffer(AT_inroughness_stat, _xy_), AT_inroughness_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_inroughness_imageSampleRect(_xy_, _dxy_)     _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_imageSampleRectClip(_xy_, _dxy_) AT_inroughness_bufferSampleRectClip(imageToBuffer(AT_inroughness_stat, _xy_), AT_inroughness_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_inroughness_imageSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(_xy_, _dxy_, _bound_inroughness).x
#endif
#ifdef HAS_inroughness
#define AT_inroughness_textureSampleRect(_xy_, _dxy_)   AT_inroughness_bufferSampleRect(textureToBuffer(AT_inroughness_stat, _xy_), (float2)(AT_inroughness_stat->resolution.x, AT_inroughness_stat->resolution.y) * (_dxy_))
#else
#define AT_inroughness_textureSampleRect(_xy_, _dxy_)   _bound_inroughness
#endif
#ifdef HAS_inroughness
#define AT_inroughness_textureSampleRectClip(_xy_, _dxy_)       AT_inroughness_bufferSampleRectClip(textureToBuffer(AT_inroughness_stat, _xy_), (float2)(AT_inroughness_stat->resolution.x, AT_inroughness_stat->resolution.y) * (_dxy_))
#else
#define AT_inroughness_textureSampleRectClip(_xy_, _dxy_)       constImageSampleRectClip(bufferToImage(AT_inroughness_stat, textureToBuffer(AT_inroughness_stat, _xy_)), _dxy_ * ((float2)(AT_inroughness_stat->resolution.x, AT_inroughness_stat->resolution.y)) * AT_inroughness_stat->buffer_to_image.lo, _bound_inroughness).x
#endif
#ifdef HAS_inroughness
#define AT_inroughness_bufferToImage(_xy_)      (bufferToImage(AT_inroughness_stat, _xy_))
#else
#define AT_inroughness_bufferToImage(_xy_)      (_xy_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_imageToBuffer(_xy_)      (imageToBuffer(AT_inroughness_stat, _xy_))
#else
#define AT_inroughness_imageToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_bufferToPixel(_xy_)      (bufferToPixel(AT_inroughness_stat, _xy_))
#else
#define AT_inroughness_bufferToPixel(_xy_)      (_xy_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_pixelToBuffer(_xy_)      (pixelToBuffer(AT_inroughness_stat, _xy_))
#else
#define AT_inroughness_pixelToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_bufferToTexture(_xy_)    (bufferToTexture(AT_inroughness_stat, _xy_))
#else
#define AT_inroughness_bufferToTexture(_xy_)    (_xy_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_textureToBuffer(_xy_)    (textureToBuffer(AT_inroughness_stat, _xy_))
#else
#define AT_inroughness_textureToBuffer(_xy_)    (_xy_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_imageToWorld(_xy_)       (imageToWorld(AT_inroughness_stat, _xy_))
#else
#define AT_inroughness_imageToWorld(_xy_)       ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_inroughness
#define AT_inroughness_image3ToWorld(_xyz_)     (image3ToWorld(AT_inroughness_stat, _xyz_))
#else
#define AT_inroughness_image3ToWorld(_xyz_)     (_xyz_)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_worldToImage(_xyz_)      (worldToImage(AT_inroughness_stat, _xyz_))
#else
#define AT_inroughness_worldToImage(_xyz_)      ((_xyz_).xy)
#endif
#ifdef HAS_inroughness
#define AT_inroughness_worldToImage3(_xyz_)     (worldToImage3(AT_inroughness_stat, _xyz_))
#else
#define AT_inroughness_worldToImage3(_xyz_)     (_xyz_)
#endif
#line 1























 










 




























// Post Processing
/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    postprocessing.h ( CE Library, OpenCL)
 *
 * COMMENTS:
 */

#ifndef __POSTPROCESSING_H__
#define __POSTPROCESSING_H__


#define DECLARE_CONTRAST_FUNC(TYPE, COMPONENTS) \
static TYPE \
post_contrast_ ## COMPONENTS (TYPE x, float contrast_exponent)\
{ \
    return (x * (1.0f + contrast_exponent)) - (contrast_exponent / 2.0f); \
}
DECLARE_CONTRAST_FUNC(float, 1)
DECLARE_CONTRAST_FUNC(float2, 2)
DECLARE_CONTRAST_FUNC(float3, 3)
DECLARE_CONTRAST_FUNC(float4, 4)

// OpenCL port of the Bias VOP
#define DECLARE_VOP_BIAS_FUNC(TYPE, COMPONENTS) \
static TYPE \
post_vop_bias_ ## COMPONENTS(TYPE x, float bias_exponent) \
{\
    return (x <= 0) ? 0.0f : ((x >= 1.0f) ? 1.0f : \
        (bias_exponent / (((1.0f / x) - 2.0f) * (1.0f - bias_exponent) + 1.0f))); \
}
DECLARE_VOP_BIAS_FUNC(float, 1)
DECLARE_VOP_BIAS_FUNC(float2, 2)
DECLARE_VOP_BIAS_FUNC(float3, 3)
DECLARE_VOP_BIAS_FUNC(float4, 4)

// OpenCL port of the Gain VOP
#define DECLARE_VOP_GAIN_FUNC(TYPE, COMPONENTS) \
static TYPE \
post_vop_gain_ ## COMPONENTS(TYPE x, float gain_exponent) \
{\
    return (x < 0.5f) ? post_vop_bias_ ## COMPONENTS(2 * x, gain_exponent) * 0.5f : \
        1.0f - post_vop_bias_ ## COMPONENTS(2 * (1.0f - x), gain_exponent) * 0.5f; \
}
DECLARE_VOP_GAIN_FUNC(float, 1)
DECLARE_VOP_GAIN_FUNC(float2, 2)
DECLARE_VOP_GAIN_FUNC(float3, 3)
DECLARE_VOP_GAIN_FUNC(float4, 4)

// Code used by the Gamma COP.
#define DECLARE_COP_GAMMA_FUNC(TYPE, COMPONENTS) \
static TYPE \
post_cop_gamma_ ## COMPONENTS(TYPE x, float gamma_exponent) \
{\
    TYPE x_sign = sign(x); \
    x = fabs(x); \
    if (gamma_exponent <= 0) \
    { \
        x = select( (TYPE)(1), (TYPE)(0), isequal(x, 0.0f)); \
    } \
    else \
    { \
        x = pow(x, gamma_exponent); \
    } \
    x *= x_sign; \
    return x; \
}
DECLARE_COP_GAMMA_FUNC(float, 1)
DECLARE_COP_GAMMA_FUNC(float2, 2)
DECLARE_COP_GAMMA_FUNC(float3, 3)
DECLARE_COP_GAMMA_FUNC(float4, 4)

#define DECLARE_POSTPROCESSING_FUNC(TYPE, COMPONENTS) \
static TYPE \
post_processing_ ## COMPONENTS(TYPE noise, int dofold, int docomplement, \
                    int dobias, float bias_exponent, int dogain, float gain_exponent, \
                    int dogamma, float gamma_exponent, int docontrast, float contrast_exponent, \
                    int doclampmin, float minimum, int doclampmax, float maximum) \
{ \
    if(dofold) \
        noise = fabs(noise); \
    \
    if(docomplement) \
        noise = 1.0f - noise; \
    \
    if(dobias) \
        noise = post_vop_bias_ ## COMPONENTS(noise, bias_exponent); \
    \
    if(dogain) \
        noise = post_vop_gain_ ## COMPONENTS(noise, gain_exponent); \
    \
    if(dogamma) \
        noise = post_cop_gamma_ ## COMPONENTS(noise, gamma_exponent); \
    \
    if(docontrast) \
        noise = post_contrast_ ## COMPONENTS(noise, contrast_exponent); \
    \
    if(doclampmin) \
        noise = noise < minimum ? minimum : noise; \
    \
    if(doclampmax) \
        noise = noise > maximum ? maximum : noise; \
    \
    return noise; \
}
DECLARE_POSTPROCESSING_FUNC(float, 1)
DECLARE_POSTPROCESSING_FUNC(float2, 2)
DECLARE_POSTPROCESSING_FUNC(float3, 3)
DECLARE_POSTPROCESSING_FUNC(float4, 4)

#endif






































// Noise Types
#define NOISE_TORUS 0
#define NOISE_PERLIN 1
#define NOISE_WORLEYFA 2
#define NOISE_WORLEYFB 3
#define NOISE_WHITE 4
#define NOISE_ALLIGATOR 5

#if NOISETYPE == NOISE_TORUS
#ifndef __RANDOM_H
#define __RANDOM_H

/******************************************************************************
 * HDK-consistent floor, integer hash, and fast random number generation.
 ******************************************************************************/

/// Returns the largest representable integer no greater than the given input.
static float
SYSfloorIL(float val)
{
    uint tmp = as_uint(val);
    uint shift = (tmp >> 23) & 0xff;

    if(shift < 0x7f)
    {
        return (tmp > 0x80000000) ? -1.0F : 0.0F;
    }
    else if(shift < 0x96)
    {
        uint mask = 0xffffffff << (0x96 - shift);
        if(tmp & 0x80000000)
        {
            if((tmp & ~mask) & 0x7fffff)
            {
                tmp &= mask;
                return as_float(tmp) - 1;
            }
            else
            {
                return val;
            }
        }
        else
        {
            return as_float(tmp & mask);
        }
    }
    else
    {
        return val;
    }
}

/// Consistent integer hash with the HDK.
static uint
SYSwang_inthash(uint key)
{
    key += ~(key << 16);
    key ^=  (key >>  5);
    key +=  (key <<  3);
    key ^=  (key >> 13);
    key += ~(key <<  9);
    key ^=  (key >> 17);
    return key;
}

/// Generates a uniform random number in the 0-1 range from the given seed and
/// updates the seed.
static float
SYSfastRandom(uint* seed)
{
    uint temp;
    *seed = (*seed) * 1664525 + 1013904223;
    temp = 0x3f800000 | (0x007fffff & (*seed));
    return as_float(temp) - 1.0f;
}

/******************************************************************************
 * Hashing macros for varying number of inputs.
 ******************************************************************************/

/// Conversion from a single float to an integer, for the clamped and unclamped
/// cases.
#define C_HASH1(x) ((int) SYSfloorIL(x))
#define U_HASH1(x) (as_uint(x))

/// Hash functions for 2-4 integers, used by clamped float hashing.
#define HASH2I(x, y) ((x^0xffffdead) * (y^0xffffc0de))
#define HASH3I(x, y, z) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169))
#define HASH4I(x, y, z, w) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169) \
                                           * (w^0xffff0397))
/// These macros declare hashing functions for clamped floating point numbers.
#define C_HASH2(x, y) HASH2I(C_HASH1(x), C_HASH1(y))
#define C_HASH3(x, y, z) HASH3I(C_HASH1(x), C_HASH1(y), C_HASH1(z))
#define C_HASH4(x, y, z, w) HASH4I(C_HASH1(x), C_HASH1(y), C_HASH1(z), \
                                   C_HASH1(w))

/// This macro hashes 2 integers, used by generic float hashing.
#define U_HASH2_RAW(x, y) y + SYSwang_inthash(x)
/// These macros declare hashing functions for generic floating point numbers.
#define U_HASH2(x, y) U_HASH2_RAW(U_HASH1(x), U_HASH1(y))
#define U_HASH3(x, y, z) U_HASH2_RAW(U_HASH2(x, y), U_HASH1(z))
#define U_HASH4(x, y, z, w) U_HASH2_RAW(U_HASH3(x, y, z), U_HASH1(w))

/******************************************************************************
 * Helper macros to hash inputs and generate outputs.
 ******************************************************************************/

/// Macros to hash the given number of floating point variables and store the
/// result in a new integer called hash.
#define C_1_HASH uint hash = SYSwang_inthash(C_HASH1(x));
#define C_2_HASH uint hash = SYSwang_inthash(C_HASH2(x, y));
#define C_3_HASH uint hash = SYSwang_inthash(C_HASH3(x, y, z));
#define C_4_HASH uint hash = SYSwang_inthash(C_HASH4(x, y, z, w));
#define U_1_HASH int hash = SYSwang_inthash(U_HASH1(x));
#define U_2_HASH int hash = SYSwang_inthash(U_HASH2(x, y));
#define U_3_HASH int hash = SYSwang_inthash(U_HASH3(x, y, z));
#define U_4_HASH int hash = SYSwang_inthash(U_HASH4(x, y, z, w));
/// Macros to return a vector of K random numbers; hash integer variable must be
/// declared and initialized.
#define RET_1_RAND return SYSfastRandom((uint*) &hash);
#define RET_2_RAND return (float2)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_3_RAND return (float3)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_4_RAND return (float4)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));

/******************************************************************************
 * Generation of final VEX_equivalent random_fhash() functions.
 ******************************************************************************/

static int VEXrandom_fhash_1(float x)
{
    U_1_HASH
    return hash;
}
static int VEXrandom_fhash_2(float x, float y)
{
    U_2_HASH
    return hash;
}
static int VEXrandom_fhash_3(float x, float y, float z)
{
    U_3_HASH
    return hash;
}
static int VEXrandom_fhash_4(float x, float y, float z, float w)
{
    U_4_HASH
    return hash;
}

/******************************************************************************
 * Generation of the final VEX-equivalent random() functions with float inputs.
 ******************************************************************************/

/// Macro that generates the code for random functions. NUM should be 2-4
/// (number of random numbers to return).
#define CREATE_RANDOM(NUM) \
static float ## NUM VEXrandom_1_ ## NUM(float x) \
{ \
    C_1_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_2_ ## NUM(float x, float y) \
{ \
    C_2_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_3_ ## NUM(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_4_ ## NUM(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point random.
#define CREATE_RANDOM_FLOAT \
static float VEXrandom_1_1(float x) \
{ \
    C_1_HASH \
    RET_1_RAND \
} \
static float VEXrandom_2_1(float x, float y) \
{ \
    C_2_HASH \
    RET_1_RAND \
} \
static float VEXrandom_3_1(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_1_RAND \
} \
static float VEXrandom_4_1(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RANDOM_FLOAT
CREATE_RANDOM(2)
CREATE_RANDOM(3)
CREATE_RANDOM(4)

/******************************************************************************
 * Generation of the final VEX-equivalent rand() functions.
 ******************************************************************************/

/// Macro that generates the code for rand functions. NUM should be 2-4 (number
/// of random numbers to return).
#define CREATE_RAND(NUM) \
static float ## NUM VEXrand_1_ ## NUM(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_2_ ## NUM(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_3_ ## NUM(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_4_ ## NUM(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point rand.
#define CREATE_RAND_FLOAT \
static float VEXrand_1_1(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_2_1(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_3_1(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_4_1(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RAND_FLOAT
CREATE_RAND(2)
CREATE_RAND(3)
CREATE_RAND(4)

#endif

#ifndef __SIMPLEX_NOISE_H
#define __SIMPLEX_NOISE_H

// #include "random.h"

/******************************************************************************
 * Declaration of constants.
 ******************************************************************************/

// Sorting permutation for each comparison bit string. The string consist of
// results of c2<c3, c1<c3, c0<c3, c1<c2, c0<c2, c0<c1.
constant int _SIMPLEX_TRAVERSAL_ORDER[64][4] =
{
    {0, 1, 2, 3},   // ABCD -> 000000 = 00
    {1, 0, 2, 3},   // BACD -> 000001 = 01
    {0, 0, 0, 0},
    {1, 2, 0, 3},   // BCAD -> 000011 = 03
    {0, 2, 1, 3},   // ACBD -> 000100 = 04
    {0, 0, 0, 0},
    {2, 0, 1, 3},   // CABD -> 000110 = 06
    {2, 1, 0, 3},   // CBAD -> 000111 = 07
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {1, 2, 3, 0},   // BCDA -> 001011 = 11
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {2, 1, 3, 0},   // CBDA -> 001111 = 15
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 2, 3, 1},   // ACDB -> 010100 = 20
    {0, 0, 0, 0},
    {2, 0, 3, 1},   // CADB -> 010110 = 22
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {2, 3, 0, 1},   // CDAB -> 011110 = 30
    {2, 3, 1, 0},   // CDBA -> 011111 = 31
    {0, 1, 3, 2},   // ABDC -> 100000 = 32
    {1, 0, 3, 2},   // BADC -> 100001 = 33
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {1, 3, 0, 2},   // BDAC -> 101001 = 41
    {0, 0, 0, 0},
    {1, 3, 2, 0},   // BDCA -> 101011 = 43
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 3, 1, 2},   // ADBC -> 110000 = 48
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 3, 2, 1},   // ADCB -> 110100 = 52
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {3, 0, 1, 2},   // DABC -> 111000 = 56
    {3, 1, 0, 2},   // DBAC -> 111001 = 57
    {0, 0, 0, 0},
    {3, 1, 2, 0},   // DBCA -> 111011 = 59
    {3, 0, 2, 1},   // DACB -> 111100 = 60
    {0, 0, 0, 0},
    {3, 2, 0, 1},   // DCAB -> 111110 = 62
    {3, 2, 1, 0}    // DCBA -> 111111 = 63
};

/// Arrays of literals for offsets in each dimensionality.
constant float _SIMPLEX_OFFSETS2[] =
{
    1.0f, 0.0f,
    0.0f, 1.0f
};
constant float _SIMPLEX_OFFSETS3[] =
{
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f
};
constant float _SIMPLEX_OFFSETS4[] =
{
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

/// Skew and unskew factors.
constant float _SIMPLEX_F[] =
{
    0.366025403784439,
    0.333333333333333,
    0.309016994374947
};
constant float _SIMPLEX_G[] =
{
    0.211324865405187,
    0.166666666666667,
    0.138196601125011
};

/******************************************************************************
 * Gradient vector code.
 ******************************************************************************/

constant float _SIMPLEX_GRADIENTS_2[] =
{
    1.0f, 0.0f,
    -1.0f, 0.0f,
    0.0f, 1.0f,
    0.0f, -1.0f,
    1.0f, 1.0f,
    -1.0f, 1.0f,
    1.0f, -1.0f,
    -1.0f, -1.0f
};
#define _SIMPLEX_NG_2 8

constant float _SIMPLEX_GRADIENTS_3[] =
{
    1.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,
    1.0f, 0.0f, 1.0f,
    -1.0f, 0.0f, 1.0f,
    1.0f, 0.0f, -1.0f,
    -1.0f, 0.0f, -1.0f,
    0.0f, 1.0f, 1.0f,
    0.0f, -1.0f, 1.0f,
    0.0f, 1.0f, -1.0f,
    0.0f, -1.0f, -1.0f
};
#define _SIMPLEX_NG_3 12

constant float _SIMPLEX_GRADIENTS_4[] =
{
    0.0f, 1.0f, 1.0f, 1.0f,
    0.0f, 1.0f, 1.0f, -1.0f,
    0.0f, 1.0f, -1.0f, 1.0f,
    0.0f, 1.0f, -1.0f, -1.0f,
    0.0f, -1.0f, 1.0f, 1.0f,
    0.0f, -1.0f, 1.0f, -1.0f,
    0.0f, -1.0f, -1.0f, 1.0f,
    0.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 0.0f, 1.0f, 1.0f,
    -1.0f, 0.0f, 1.0f, 1.0f,
    1.0f, 0.0f, 1.0f, -1.0f,
    -1.0f, 0.0f, 1.0f, -1.0f,
    1.0f, 0.0f, -1.0f, 1.0f,
    -1.0f, 0.0f, -1.0f, 1.0f,
    1.0f, 0.0f, -1.0f, -1.0f,
    -1.0f, 0.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, -1.0f, 0.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f, -1.0f,
    1.0f, -1.0f, 0.0f, -1.0f,
    -1.0f, 1.0f, 0.0f, -1.0f,
    -1.0f, -1.0f, 0.0f, -1.0f,
    1.0f, 1.0f, 1.0f, 0.0f,
    1.0f, 1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 1.0f, 0.0f,
    1.0f, -1.0f, -1.0f, 0.0f,
    -1.0f, 1.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, -1.0f, 0.0f,
    -1.0f, -1.0f, 1.0f, 0.0f,
    -1.0f, -1.0f, -1.0f, 0.0f
};
#define _SIMPLEX_NG_4 32

/// Return gradient vector at each cell, given the seed.
float2 _getGradient_2(float2 x, uint seed)
{
    uint idx = as_uint(x.y) + SYSwang_inthash(as_uint(x.x));
    idx = SYSwang_inthash(seed + SYSwang_inthash(idx));
    return vload2(idx % _SIMPLEX_NG_2, _SIMPLEX_GRADIENTS_2);
}
float3 _getGradient_3(float3 x, uint seed)
{
    uint idx = as_uint(x.y) + SYSwang_inthash(as_uint(x.x));
    idx = as_uint(x.z) + SYSwang_inthash(idx);
    idx = SYSwang_inthash(seed + SYSwang_inthash(idx));
    return vload3(idx % _SIMPLEX_NG_3, _SIMPLEX_GRADIENTS_3);
}
float4 _getGradient_4(float4 x, uint seed)
{
    uint idx = as_uint(x.y) + SYSwang_inthash(as_uint(x.x));
    idx = as_uint(x.z) + SYSwang_inthash(idx);
    idx = as_uint(x.w) + SYSwang_inthash(idx);
    idx = SYSwang_inthash(seed + SYSwang_inthash(idx));
    return vload4(idx % _SIMPLEX_NG_4, _SIMPLEX_GRADIENTS_4);
}

/******************************************************************************
 * Function declaration templates.
 ******************************************************************************/

/// __SEED_X are the gradient seeds for noise of output dimension X.
#define __SEED_1(IN_DIM) IN_DIM
#define __SEED_2(IN_DIM) (uint2)(5, 10) + 10 * IN_DIM
#define __SEED_3(IN_DIM) (uint3)(10, 22, 34) + 100 * IN_DIM
#define __SEED_4(IN_DIM) (uint4)(20, 40, 60, 80) + 1000 * IN_DIM

/// __SUM_X(v) is the sum of all components of the X-dimensional vector v.
#define __SUM_1(vector) (vector)
#define __SUM_2(vector) (vector.x + vector.y)
#define __SUM_3(vector) (vector.x + vector.y + vector.z)
#define __SUM_4(vector) (vector.x + vector.y + vector.z + vector.w)

/// Vector (and seed) type for each dimensionality.
#define __OP_1 float
#define __OP_2 float2
#define __OP_3 float3
#define __OP_4 float4
#define __SEED_OP_1 uint
#define __SEED_OP_2 uint2
#define __SEED_OP_3 uint3
#define __SEED_OP_4 uint4

/// Amplification factors.
#define __AMP_2 35.0f
#define __AMP_3 38.4f
#define __AMP_4 31.3f

/// Position adjustment factors.
#define __FREQ_ADJ 0.65f

/// Gives the pairwise comparison score for the passed vector.
#define __PAIRWISE_COMP_SCORE_2(vector) \
    (vector.x < vector.y ? 1 : 0)
#define __PAIRWISE_COMP_SCORE_3(vector) \
    (vector.y < vector.z ? 4 : 0) + \
    (vector.x < vector.z ? 2 : 0) + \
    __PAIRWISE_COMP_SCORE_2(vector)
#define __PAIRWISE_COMP_SCORE_4(vector) \
    (vector.z < vector.w ? 32 : 0) + \
    (vector.y < vector.w ? 16 : 0) + \
    (vector.x < vector.w ? 8 : 0) + \
    __PAIRWISE_COMP_SCORE_3(vector)

/// __STACKED_GRAD_DOTS_X(Y) is the stacked vector of dot products of disp and
/// the pseudorandom gradient vector _getGradient_Y(int_x, seeds).
#define __SINGLE_GRAD_DOT(IN_DIM, SEED) \
    dot(disp, _getGradient_ ## IN_DIM (int_x, SEED))
#define __STACKED_GRAD_DOTS_1(IN_DIM) \
    __SINGLE_GRAD_DOT(IN_DIM, seed)
#define __STACKED_GRAD_DOTS_2(IN_DIM) \
    (float2)(__SINGLE_GRAD_DOT(IN_DIM, seed.x), \
             __SINGLE_GRAD_DOT(IN_DIM, seed.y))
#define __STACKED_GRAD_DOTS_3(IN_DIM) \
    (float3)(__SINGLE_GRAD_DOT(IN_DIM, seed.x), \
             __SINGLE_GRAD_DOT(IN_DIM, seed.y), \
             __SINGLE_GRAD_DOT(IN_DIM, seed.z))
#define __STACKED_GRAD_DOTS_4(IN_DIM) \
    (float4)(__SINGLE_GRAD_DOT(IN_DIM, seed.x), \
             __SINGLE_GRAD_DOT(IN_DIM, seed.y), \
             __SINGLE_GRAD_DOT(IN_DIM, seed.z), \
             __SINGLE_GRAD_DOT(IN_DIM, seed.w))

/// Declares the function of the form
///     floatY VEXgxnoise_X_Y(floatX x)
/// where X, Y are IN_DIM, OUT_DIM, respectively.
#define __DECLARE_NOISE_FUNC(IN_DIM, OUT_DIM) \
__OP_ ## OUT_DIM VEXgxnoise_ ## IN_DIM ## _ ## OUT_DIM (__OP_ ## IN_DIM x) \
{ \
    __SEED_OP_ ## OUT_DIM seed = __SEED_ ## OUT_DIM (IN_DIM);

/// Declares the helper function of the form
///     floatX __gxnoise_deriv_H_X(floatX x, uint seed)
#define __DECLARE_NOISE_DERIV_FUNC(IN_OUT_DIM) \
__OP_ ## IN_OUT_DIM __gxnoise_deriv_H_ ## IN_OUT_DIM (__OP_ ## IN_OUT_DIM x, \
    uint seed) \
{ 

/// Start of the noise function logic. The coordinates are first skewed, and the
/// corresponding cell located. This cell is then unskewed, and from it disp is
/// calculated (offset between the input and the cell origin). Finally, c is
/// computed--this is the traversal order index in the array of permutations.
#define __NOISE_FUNC_START(IN_DIM) \
    x *= __FREQ_ADJ; \
    float t = __SUM_ ## IN_DIM (x) * _SIMPLEX_F[IN_DIM - 2]; \
    __OP_ ## IN_DIM int_x = floor(x + t); \
    t = __SUM_ ## IN_DIM (int_x) * _SIMPLEX_G[IN_DIM - 2]; \
    __OP_ ## IN_DIM disp = x - (int_x - t); \
    int c = __PAIRWISE_COMP_SCORE_ ## IN_DIM (disp);

/// Adds the kernel contribution to val.
#define __NOISE_UPDATE(IN_DIM, OUT_DIM) \
    t = 0.5f - dot(disp, disp); \
    if (t > 0) \
    { \
        t = t * t; \
        val += t * t * __STACKED_GRAD_DOTS_ ## OUT_DIM (IN_DIM); \
    }

/// Adds the kernel derivative contribution to dest for the current grad and
/// disp vectors (and the up-to-date t value).
#define __RAW_DERIV_UPDATE(dest) \
        dest += (t * t * t) * (t * grad - 8.0f * dot(disp, grad) * disp);

/// Adds the kernel derivative contribution to val.
#define __DERIV_UPDATE(IN_OUT_DIM) \
    t = 0.5f - dot(disp, disp); \
    if (t > 0) \
    { \
        __OP_ ## IN_OUT_DIM grad = _getGradient_ ## IN_OUT_DIM(int_x, seed); \
        __RAW_DERIV_UPDATE(val) \
    }

/// Adds the kernel derivative contributions to deriv1, deriv2, and deriv3.
#define __DERIV_UPDATE3(IN_DIM) \
    t = 0.5f - dot(disp, disp); \
    if (t > 0) \
    { \
        __OP_ ## IN_DIM grad = _getGradient_ ## IN_DIM(int_x, seed.x); \
        __RAW_DERIV_UPDATE(deriv1); \
        grad = _getGradient_ ## IN_DIM(int_x, seed.y); \
        __RAW_DERIV_UPDATE(deriv2); \
        grad = _getGradient_ ## IN_DIM(int_x, seed.z); \
        __RAW_DERIV_UPDATE(deriv3); \
    }

/// Evaluates to the appropriate offset vector.
#define __SET_SIMPLEX_OFFSET(IN_DIM) \
    __OP_ ## IN_DIM offset = vload ## IN_DIM (_SIMPLEX_TRAVERSAL_ORDER[c][i], \
                                              _SIMPLEX_OFFSETS ## IN_DIM);


/// Updates vertex coordinate and the offset for the current iteration.
#define __CELL_DISP_UPDATE(IN_DIM) \
        __SET_SIMPLEX_OFFSET(IN_DIM) \
        int_x += offset; \
        disp -= offset - _SIMPLEX_G[IN_DIM - 2];

/// Computes the noise value in val.
#define __NOISE_VAL_COMPUTE(IN_DIM, OUT_DIM) \
    __OP_ ## OUT_DIM val = 0; \
    __NOISE_UPDATE(IN_DIM, OUT_DIM) \
    for (int i = 0; i < IN_DIM; i++) \
    { \
        __CELL_DISP_UPDATE(IN_DIM) \
        __NOISE_UPDATE(IN_DIM, OUT_DIM) \
    }

/// Computes gradient of the noise value in val.
#define __NOISE_DERIV_COMPUTE(IN_OUT_DIM) \
    __OP_ ## IN_OUT_DIM val = 0; \
    __DERIV_UPDATE(IN_OUT_DIM) \
    for (int i = 0; i < IN_OUT_DIM; i++) \
    { \
        __CELL_DISP_UPDATE(IN_OUT_DIM) \
        __DERIV_UPDATE(IN_OUT_DIM) \
    }

/// Computes gradients of the first three noise components in deriv1, deriv2,
/// deriv3 vectors.
#define __NOISE_DERIV_COMPUTE3(IN_DIM) \
    __OP_ ## IN_DIM deriv1 = 0; \
    __OP_ ## IN_DIM deriv2 = 0; \
    __OP_ ## IN_DIM deriv3 = 0; \
    __DERIV_UPDATE3(IN_DIM) \
    for (int i = 0; i < IN_DIM; i++) \
    { \
        __CELL_DISP_UPDATE(IN_DIM) \
        __DERIV_UPDATE3(IN_DIM) \
    }

/// Returns val and closes the function block (for the noise itself).
#define __NOISE_FUNC_END(IN_DIM) \
    return val * __AMP_ ## IN_DIM + 0.5f; \
}

/// Returns val and closes the function block (for the noise derivative).
#define __NOISE_DERIV_FUNC_END(IN_DIM) \
    return val * (__AMP_ ## IN_DIM * __FREQ_ADJ); \
}

/// Creates the code for the VEXgxnoise_X_Y() function.
#define __CREATE_NOISE_FUNC(IN_DIM, OUT_DIM) \
__DECLARE_NOISE_FUNC(IN_DIM, OUT_DIM) \
__NOISE_FUNC_START(IN_DIM) \
__NOISE_VAL_COMPUTE(IN_DIM, OUT_DIM) \
__NOISE_FUNC_END(IN_DIM)

/// Creates the code for the __gxnoise_deriv_H_X() helper function.
#define __CREATE_NOISE_DERIV_FUNC(IN_OUT_DIM) \
__DECLARE_NOISE_DERIV_FUNC(IN_OUT_DIM) \
__NOISE_FUNC_START(IN_OUT_DIM) \
__NOISE_DERIV_COMPUTE(IN_OUT_DIM) \
__NOISE_DERIV_FUNC_END(IN_OUT_DIM)

/// Creates all output noise flavors for the given input dimensionality.
#define CREATE_NOISE_FOR_ALL_OUTPUTS(IN_DIM) \
__CREATE_NOISE_FUNC(IN_DIM, 1) \
__CREATE_NOISE_FUNC(IN_DIM, 2) \
__CREATE_NOISE_FUNC(IN_DIM, 3) \
__CREATE_NOISE_FUNC(IN_DIM, 4) \

/// Creates the code for the sx_curl_noise_X_2() function.
#define CREATE_CURL_NOISE_FUNC2(IN_DIM) \
__CREATE_NOISE_DERIV_FUNC(IN_DIM) \
float2 VEXcurlgxnoise2d_ ## IN_DIM(__OP_ ## IN_DIM x) \
{ \
    __OP_ ## IN_DIM deriv \
        = __gxnoise_deriv_H_ ## IN_DIM (x, __SEED_1(IN_DIM)); \
    return (float2)(deriv.y, -deriv.x); \
}

/// Creates the code for the sx_curl_noise_X_3() function.
#define CREATE_CURL_NOISE_FUNC3(IN_DIM) \
float3 VEXcurlgxnoise_ ## IN_DIM(__OP_ ## IN_DIM x) \
{ \
    uint3 seed = __SEED_3(IN_DIM); \
    __NOISE_FUNC_START(IN_DIM) \
    __NOISE_DERIV_COMPUTE3(IN_DIM) \
    return (float3)(deriv3.y - deriv2.z, \
                    deriv1.z - deriv3.x, \
                    deriv2.x - deriv1.y) \
        * (__AMP_ ## IN_DIM * __FREQ_ADJ); \
}

/******************************************************************************
 * Function definitions.
 ******************************************************************************/

/// Actually creates the 12 noise functions
///     floatO VEXgxnoise_I_O(floatI x)
/// for O in {1, 2, 3, 4}, and I in {2, 3, 4} (float1 is float).
CREATE_NOISE_FOR_ALL_OUTPUTS(2)
CREATE_NOISE_FOR_ALL_OUTPUTS(3)
CREATE_NOISE_FOR_ALL_OUTPUTS(4)

/// Actually creates the functions
///     float2 VEXcurlgxnoise2d_I(floatI x)
/// for I in {2, 3, 4}.
CREATE_CURL_NOISE_FUNC2(2)
CREATE_CURL_NOISE_FUNC2(3)
CREATE_CURL_NOISE_FUNC2(4)

/// Actually creates the functions
///     float3 VEXcurlgxnoise_I(floatI x)
/// for I in {3, 4}.
CREATE_CURL_NOISE_FUNC3(3)
CREATE_CURL_NOISE_FUNC3(4)

#endif

#elif NOISETYPE == NOISE_ALLIGATOR
#ifndef __RANDOM_H
#define __RANDOM_H

/******************************************************************************
 * HDK-consistent floor, integer hash, and fast random number generation.
 ******************************************************************************/

/// Returns the largest representable integer no greater than the given input.
static float
SYSfloorIL(float val)
{
    uint tmp = as_uint(val);
    uint shift = (tmp >> 23) & 0xff;

    if(shift < 0x7f)
    {
        return (tmp > 0x80000000) ? -1.0F : 0.0F;
    }
    else if(shift < 0x96)
    {
        uint mask = 0xffffffff << (0x96 - shift);
        if(tmp & 0x80000000)
        {
            if((tmp & ~mask) & 0x7fffff)
            {
                tmp &= mask;
                return as_float(tmp) - 1;
            }
            else
            {
                return val;
            }
        }
        else
        {
            return as_float(tmp & mask);
        }
    }
    else
    {
        return val;
    }
}

/// Consistent integer hash with the HDK.
static uint
SYSwang_inthash(uint key)
{
    key += ~(key << 16);
    key ^=  (key >>  5);
    key +=  (key <<  3);
    key ^=  (key >> 13);
    key += ~(key <<  9);
    key ^=  (key >> 17);
    return key;
}

/// Generates a uniform random number in the 0-1 range from the given seed and
/// updates the seed.
static float
SYSfastRandom(uint* seed)
{
    uint temp;
    *seed = (*seed) * 1664525 + 1013904223;
    temp = 0x3f800000 | (0x007fffff & (*seed));
    return as_float(temp) - 1.0f;
}

/******************************************************************************
 * Hashing macros for varying number of inputs.
 ******************************************************************************/

/// Conversion from a single float to an integer, for the clamped and unclamped
/// cases.
#define C_HASH1(x) ((int) SYSfloorIL(x))
#define U_HASH1(x) (as_uint(x))

/// Hash functions for 2-4 integers, used by clamped float hashing.
#define HASH2I(x, y) ((x^0xffffdead) * (y^0xffffc0de))
#define HASH3I(x, y, z) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169))
#define HASH4I(x, y, z, w) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169) \
                                           * (w^0xffff0397))
/// These macros declare hashing functions for clamped floating point numbers.
#define C_HASH2(x, y) HASH2I(C_HASH1(x), C_HASH1(y))
#define C_HASH3(x, y, z) HASH3I(C_HASH1(x), C_HASH1(y), C_HASH1(z))
#define C_HASH4(x, y, z, w) HASH4I(C_HASH1(x), C_HASH1(y), C_HASH1(z), \
                                   C_HASH1(w))

/// This macro hashes 2 integers, used by generic float hashing.
#define U_HASH2_RAW(x, y) y + SYSwang_inthash(x)
/// These macros declare hashing functions for generic floating point numbers.
#define U_HASH2(x, y) U_HASH2_RAW(U_HASH1(x), U_HASH1(y))
#define U_HASH3(x, y, z) U_HASH2_RAW(U_HASH2(x, y), U_HASH1(z))
#define U_HASH4(x, y, z, w) U_HASH2_RAW(U_HASH3(x, y, z), U_HASH1(w))

/******************************************************************************
 * Helper macros to hash inputs and generate outputs.
 ******************************************************************************/

/// Macros to hash the given number of floating point variables and store the
/// result in a new integer called hash.
#define C_1_HASH uint hash = SYSwang_inthash(C_HASH1(x));
#define C_2_HASH uint hash = SYSwang_inthash(C_HASH2(x, y));
#define C_3_HASH uint hash = SYSwang_inthash(C_HASH3(x, y, z));
#define C_4_HASH uint hash = SYSwang_inthash(C_HASH4(x, y, z, w));
#define U_1_HASH int hash = SYSwang_inthash(U_HASH1(x));
#define U_2_HASH int hash = SYSwang_inthash(U_HASH2(x, y));
#define U_3_HASH int hash = SYSwang_inthash(U_HASH3(x, y, z));
#define U_4_HASH int hash = SYSwang_inthash(U_HASH4(x, y, z, w));
/// Macros to return a vector of K random numbers; hash integer variable must be
/// declared and initialized.
#define RET_1_RAND return SYSfastRandom((uint*) &hash);
#define RET_2_RAND return (float2)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_3_RAND return (float3)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_4_RAND return (float4)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));

/******************************************************************************
 * Generation of final VEX_equivalent random_fhash() functions.
 ******************************************************************************/

static int VEXrandom_fhash_1(float x)
{
    U_1_HASH
    return hash;
}
static int VEXrandom_fhash_2(float x, float y)
{
    U_2_HASH
    return hash;
}
static int VEXrandom_fhash_3(float x, float y, float z)
{
    U_3_HASH
    return hash;
}
static int VEXrandom_fhash_4(float x, float y, float z, float w)
{
    U_4_HASH
    return hash;
}

/******************************************************************************
 * Generation of the final VEX-equivalent random() functions with float inputs.
 ******************************************************************************/

/// Macro that generates the code for random functions. NUM should be 2-4
/// (number of random numbers to return).
#define CREATE_RANDOM(NUM) \
static float ## NUM VEXrandom_1_ ## NUM(float x) \
{ \
    C_1_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_2_ ## NUM(float x, float y) \
{ \
    C_2_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_3_ ## NUM(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_4_ ## NUM(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point random.
#define CREATE_RANDOM_FLOAT \
static float VEXrandom_1_1(float x) \
{ \
    C_1_HASH \
    RET_1_RAND \
} \
static float VEXrandom_2_1(float x, float y) \
{ \
    C_2_HASH \
    RET_1_RAND \
} \
static float VEXrandom_3_1(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_1_RAND \
} \
static float VEXrandom_4_1(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RANDOM_FLOAT
CREATE_RANDOM(2)
CREATE_RANDOM(3)
CREATE_RANDOM(4)

/******************************************************************************
 * Generation of the final VEX-equivalent rand() functions.
 ******************************************************************************/

/// Macro that generates the code for rand functions. NUM should be 2-4 (number
/// of random numbers to return).
#define CREATE_RAND(NUM) \
static float ## NUM VEXrand_1_ ## NUM(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_2_ ## NUM(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_3_ ## NUM(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_4_ ## NUM(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point rand.
#define CREATE_RAND_FLOAT \
static float VEXrand_1_1(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_2_1(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_3_1(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_4_1(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RAND_FLOAT
CREATE_RAND(2)
CREATE_RAND(3)
CREATE_RAND(4)

#endif

/*
Noise Library.

This library is a modified version of the noise library found in
Open Shading Language:
github.com/imageworks/OpenShadingLanguage/blob/master/src/include/OSL/oslnoise.h

It contains the subset of noise types needed to implement the MaterialX
standard library. The modifications are mainly conversions from C++ to GLSL.
Produced results should be identical to the OSL noise functions.

Original copyright notice:
------------------------------------------------------------------------
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
------------------------------------------------------------------------
*/

#ifndef __MX_NOISE_INTERNAL_H__
#define __MX_NOISE_INTERNAL_H__

static int
mx_wrap_int(int x, int period)
{
    if (period <= 0)
        return x;
    int y = x - x / period * period;
    return y >= 0 ? y : y + period;
}

static float
mx_select(bool b, float t, float f)
{
    return b ? t : f;
}

static float
mx_negate_if(float val, bool b)
{
    return b ? -val : val;
}

static int
mx_floor(float x)
{
    return (int)(floor(x));
}

// return mx_floor as well as the fractional remainder
static float
mx_floorfrac(float x, int* i)
{
    *i = mx_floor(x);
    return x - (float)(*i);
}

static float
mx_bilerp1(float v0, float v1, float v2, float v3, float s, float t)
{
    float s1 = 1.0f - s;
    return (1.0f - t) * (v0*s1 + v1*s) + t * (v2*s1 + v3*s);
}
static float3
mx_bilerp3(float3 v0, float3 v1, float3 v2, float3 v3,
           float s, float t)
{
    float s1 = 1.0f - s;
    return (1.0f - t) * (v0*s1 + v1*s) + t * (v2*s1 + v3*s);
}
static float
mx_trilerp1(float v0, float v1, float v2, float v3,
            float v4, float v5, float v6, float v7,
            float s, float t, float r)
{
    float s1 = 1.0f - s;
    float t1 = 1.0f - t;
    float r1 = 1.0f - r;
    return (r1*(t1*(v0*s1 + v1*s) + t*(v2*s1 + v3*s)) +
            r*(t1*(v4*s1 + v5*s) + t*(v6*s1 + v7*s)));
}
static float3
mx_trilerp3(float3 v0, float3 v1, float3 v2, float3 v3,
            float3 v4, float3 v5, float3 v6, float3 v7,
            float s, float t, float r)
{
    float s1 = 1.0f - s;
    float t1 = 1.0f - t;
    float r1 = 1.0f - r;
    return (r1*(t1*(v0*s1 + v1*s) + t*(v2*s1 + v3*s)) +
            r*(t1*(v4*s1 + v5*s) + t*(v6*s1 + v7*s)));
}

// 2 and 3 dimensional gradient functions - perform a dot product against a
// randomly chosen vector. Note that the gradient vector is not normalized, but
// this only affects the overal "scale" of the result, so we simply account for
// the scale by multiplying in the corresponding "perlin" function.
static float
mx_gradient_float1_2(uint hash, float x, float y)
{
    // 8 possible directions (+-1,+-2) and (+-2,+-1)
    uint h = hash & 7u;
    float u = mx_select(h<4u, x, y);
    float v = 2.0 * mx_select(h<4u, y, x);
    // compute the dot product with (x,y).
    return mx_negate_if(u, (bool)(h&1u)) + mx_negate_if(v, (bool)(h&2u));
}
static float
mx_gradient_float1_3(uint hash, float x, float y, float z)
{
    // use vectors pointing to the edges of the cube
    uint h = hash & 15u;
    float u = mx_select(h<8u, x, y);
    float v = mx_select(h<4u, y, mx_select((h==12u)||(h==14u), x, z));
    return mx_negate_if(u, (bool)(h&1u)) + mx_negate_if(v, (bool)(h&2u));
}
static float3
mx_gradient_float3_2(uint3 hash, float x, float y)
{
    return (float3)(mx_gradient_float1_2(hash.x, x, y),
                    mx_gradient_float1_2(hash.y, x, y),
                    mx_gradient_float1_2(hash.z, x, y));
}
static float3
mx_gradient_float3_3(uint3 hash, float x, float y, float z)
{
    return (float3)(mx_gradient_float1_3(hash.x, x, y, z),
                    mx_gradient_float1_3(hash.y, x, y, z),
                    mx_gradient_float1_3(hash.z, x, y, z));
}
// Scaling factors to normalize the result of gradients above.
// These factors were experimentally calculated to be:
//    2D:   0.6616
//    3D:   0.9820
static float mx_gradient_scale2d_1(float v) { return 0.6616f * v; }
static float mx_gradient_scale3d_1(float v) { return 0.9820f * v; }
static float3 mx_gradient_scale2d_3(float3 v) { return 0.6616f * v; }
static float3 mx_gradient_scale3d_3(float3 v) { return 0.9820f * v; }

/// Bitwise circular rotation left by k bits (for 32 bit unsigned integers)
static uint
mx_rotl32(uint x, int k)
{
    return (x<<k) | (x>>(32-k));
}

static void
mx_bjmix(uint* a, uint* b, uint* c)
{
    *a -= *c; *a ^= mx_rotl32(*c, 4); *c += *b;
    *b -= *a; *b ^= mx_rotl32(*a, 6); *a += *c;
    *c -= *b; *c ^= mx_rotl32(*b, 8); *b += *a;
    *a -= *c; *a ^= mx_rotl32(*c,16); *c += *b;
    *b -= *a; *b ^= mx_rotl32(*a,19); *a += *c;
    *c -= *b; *c ^= mx_rotl32(*b, 4); *b += *a;
}

// Mix up and combine the bits of a, b, and c (doesn't change them, but
// returns a hash of those three original values).
static uint
mx_bjfinal(uint a, uint b, uint c)
{
    c ^= b; c -= mx_rotl32(b,14);
    a ^= c; a -= mx_rotl32(c,11);
    b ^= a; b -= mx_rotl32(a,25);
    c ^= b; c -= mx_rotl32(b,16);
    a ^= c; a -= mx_rotl32(c,4);
    b ^= a; b -= mx_rotl32(a,14);
    c ^= b; c -= mx_rotl32(b,24);
    return c;
}

// Convert a 32 bit integer into a floating point number in [0,1]
static float
mx_bits_to_01(uint bits)
{
    return (float)(bits) / (float)((uint)(0xffffffff));
}

static float
mx_fade(float t)
{
   return t * t * t * (t * (t * 6.0f- 15.0f) + 10.0f);
}

static uint
mx_hash_int1p(int x, int px)
{
    x = mx_wrap_int(x, px);
    
    uint len = 1u;
    uint seed = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    return mx_bjfinal(seed+(uint)(x), seed, seed);
}

static uint
mx_hash_int2p(int x, int y, int px, int py)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);

    uint len = 2u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    return mx_bjfinal(a, b, c);
}

static uint
mx_hash_int3p(int x, int y, int z, int px, int py, int pz)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);
    
    uint len = 3u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    c += (uint)(z);
    return mx_bjfinal(a, b, c);
}

static uint
mx_hash_int4p(int x, int y, int z, int xx, int px, int py, int pz, int pxx)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);
    xx = mx_wrap_int(xx, pxx);
    
    uint len = 4u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    c += (uint)(z);
    mx_bjmix(&a, &b, &c);
    a += (uint)(xx);
    return mx_bjfinal(a, b, c);
}

static uint
mx_hash_int4pb(int x, int y, int z, int xx, int yy,
               int px, int py, int pz, int pxx)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);
    xx = mx_wrap_int(xx, pxx);

    uint len = 5u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    c += (uint)(z);
    mx_bjmix(&a, &b, &c);
    a += (uint)(xx);
    b += (uint)(yy);
    return mx_bjfinal(a, b, c);
}

static uint3
mx_hash_float3_2p(int x, int y, int px, int py)
{
    uint h = mx_hash_int2p(x, y, px, py);
    // we only need the low-order bits to be random, so split out
    // the 32 bit result into 3 parts for each channel
    uint3 result;
    result.x = (h      ) & 0xFFu;
    result.y = (h >> 8 ) & 0xFFu;
    result.z = (h >> 16) & 0xFFu;
    return result;
}

static uint3
mx_hash_float3_3p(int x, int y, int z, int px, int py, int pz)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);

    uint h = mx_hash_int3p(x, y, z, px, py, pz);
    // we only need the low-order bits to be random, so split out
    // the 32 bit result into 3 parts for each channel
    uint3 result;
    result.x = (h      ) & 0xFFu;
    result.y = (h >> 8 ) & 0xFFu;
    result.z = (h >> 16) & 0xFFu;
    return result;
}

static float
mx_perlin_noise_float_2(float2 p, int2 per)
{
    int X, Y;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float result = mx_bilerp1(
        mx_gradient_float1_2(mx_hash_int2p(X  , Y  , per.x, per.y), fx    , fy    ),
        mx_gradient_float1_2(mx_hash_int2p(X+1, Y  , per.x, per.y), fx-1.0f, fy    ),
        mx_gradient_float1_2(mx_hash_int2p(X  , Y+1, per.x, per.y), fx    , fy-1.0f),
        mx_gradient_float1_2(mx_hash_int2p(X+1, Y+1, per.x, per.y), fx-1.0f, fy-1.0f),
        u, v);
    return mx_gradient_scale2d_1(result);
}

static float
mx_perlin_noise_float_3(float3 p, int3 per)
{
    int X, Y, Z;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float fz = mx_floorfrac(p.z, &Z);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float w = mx_fade(fz);
    float result = mx_trilerp1(
        mx_gradient_float1_3(mx_hash_int3p(X  , Y  , Z  , per.x, per.y, per.z),
                             fx    , fy    , fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y  , Z  , per.x, per.y, per.z),
                             fx-1.0f, fy    , fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X  , Y+1, Z  , per.x, per.y, per.z),
                             fx    , fy-1.0f, fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y+1, Z  , per.x, per.y, per.z),
                             fx-1.0f, fy-1.0f, fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X  , Y  , Z+1, per.x, per.y, per.z),
                             fx    , fy    , fz-1.0f),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y  , Z+1, per.x, per.y, per.z),
                             fx-1.0f, fy    , fz-1.0f),
        mx_gradient_float1_3(mx_hash_int3p(X  , Y+1, Z+1, per.x, per.y, per.z),
                             fx    , fy-1.0f, fz-1.0f),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y+1, Z+1, per.x, per.y, per.z),
                             fx-1.0f, fy-1.0f, fz-1.0f),
        u, v, w);
    return mx_gradient_scale3d_1(result);
}

static float3
mx_perlin_noise_float3_2(float2 p, int2 per)
{
    int X, Y;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float3 result = mx_bilerp3(
        mx_gradient_float3_2(mx_hash_float3_2p(X  , Y  , per.x, per.y), fx    , fy    ),
        mx_gradient_float3_2(mx_hash_float3_2p(X+1, Y  , per.x, per.y), fx-1.0f, fy    ),
        mx_gradient_float3_2(mx_hash_float3_2p(X  , Y+1, per.x, per.y), fx    , fy-1.0f),
        mx_gradient_float3_2(mx_hash_float3_2p(X+1, Y+1, per.x, per.y), fx-1.0f, fy-1.0f),
        u, v);
    return mx_gradient_scale2d_3(result);
}

static float3
mx_perlin_noise_float3_3(float3 p, int3 per)
{
    int X, Y, Z;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float fz = mx_floorfrac(p.z, &Z);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float w = mx_fade(fz);
    float3 result = mx_trilerp3(
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y  , Z  , per.x, per.y, per.z),
                         fx    , fy    , fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y  , Z  , per.x, per.y, per.z),
                         fx-1.0f, fy    , fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y+1, Z  , per.x, per.y, per.z),
                         fx    , fy-1.0f, fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y+1, Z  , per.x, per.y, per.z),
                         fx-1.0f, fy-1.0f, fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y  , Z+1, per.x, per.y, per.z),
                         fx    , fy    , fz-1.0f),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y  , Z+1, per.x, per.y, per.z),
                         fx-1.0f, fy    , fz-1.0f),
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y+1, Z+1, per.x, per.y, per.z),
                         fx    , fy-1.0f, fz-1.0f),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y+1, Z+1, per.x, per.y, per.z),
                         fx-1.0f, fy-1.0f, fz-1.0f),
        u, v, w);
    return mx_gradient_scale3d_3(result);
}

static float
mx_cell_noise_float_1(float p, int per)
{
    int ix = mx_floor(p);
    return mx_bits_to_01(mx_hash_int1p(ix, per));
}

static float
mx_cell_noise_float_2(float2 p, int2 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    return mx_bits_to_01(mx_hash_int2p(ix, iy, per.x, per.y));
}

static float
mx_cell_noise_float_3(float3 p, int3 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    return mx_bits_to_01(mx_hash_int3p(ix, iy, iz, per.x, per.y, per.z));
}

static float
mx_cell_noise_float_4(float4 p, int4 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    int iw = mx_floor(p.w);
    return mx_bits_to_01(mx_hash_int4p(ix, iy, iz, iw,
                                     per.x, per.y, per.z, per.w));
}

static float3
mx_cell_noise_float3_1(float p, int per)
{
    int ix = mx_floor(p);
    return (float3)(
            mx_bits_to_01(mx_hash_int2p(ix, 0, per, 0)),
            mx_bits_to_01(mx_hash_int2p(ix, 1, per, 0)),
            mx_bits_to_01(mx_hash_int2p(ix, 2, per, 0))
    );
}

static float3
mx_cell_noise_float3_2(float2 p, int2 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    return (float3)(
            mx_bits_to_01(mx_hash_int3p(ix, iy, 0, per.x, per.y, 0)),
            mx_bits_to_01(mx_hash_int3p(ix, iy, 1, per.x, per.y, 0)),
            mx_bits_to_01(mx_hash_int3p(ix, iy, 2, per.x, per.y, 0))
    );
}

static float3
mx_cell_noise_float3_3(float3 p, int3 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    return (float3)(
            mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 0, per.x, per.y, per.z, 0)),
            mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 1, per.x, per.y, per.z, 0)),
            mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 2, per.x, per.y, per.z, 0))
    );
}

static float3
mx_cell_noise_float3_4(float4 p, int4 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    int iw = mx_floor(p.w);
    return (float3)(
            mx_bits_to_01(mx_hash_int4pb(ix, iy, iz, iw, 0,
                                         per.x, per.y, per.z, per.w)),
            mx_bits_to_01(mx_hash_int4pb(ix, iy, iz, iw, 1,
                                         per.x, per.y, per.z, per.w)),
            mx_bits_to_01(mx_hash_int4pb(ix, iy, iz, iw, 2,
                                         per.x, per.y, per.z, per.w))
    );
}

static float
mx_fractal_noise_float(float3 p, int octaves, float lacunarity, float diminish,
                       int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.
    
    float result = 0.0f;
    float amplitude = 1.0;
    for (int i = 0;  i < octaves; ++i)
    {
        result += amplitude * mx_perlin_noise_float_3(p, per);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}

static float3
mx_fractal_noise_float3(float3 p, int octaves, float lacunarity, float diminish,
                        int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.

    float3 result = (float3)(0.0f);
    float amplitude = 1.0;
    for (int i = 0;  i < octaves; ++i)
    {
        result += amplitude * mx_perlin_noise_float3_3(p, per);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}

static float2
mx_fractal_noise_float2(float3 p, int octaves, float lacunarity, float diminish,
                        int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.
    // Offsetting positions for the second component results in the exact same
    // value for both if offset is an integral vector multiple of period.

    return (float2)(mx_fractal_noise_float(p, octaves, lacunarity,
                                           diminish, per),
                    mx_fractal_noise_float(p+(float3)(19, 193, 17), octaves, lacunarity,
                                           diminish, per));
}

static float4
mx_fractal_noise_float4(float3 p, int octaves, float lacunarity, float diminish,
                        int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.

    float3 c = mx_fractal_noise_float3(p, octaves, lacunarity, diminish, per);
    float f = mx_fractal_noise_float(p+(float3)(19, 193, 17), octaves, lacunarity,
                                     diminish, per);
    return (float4)(c, f);
}

static float
mx_worley_distance2(float2 p, int x, int y, int xoff, int yoff, float jitter,
                    int metric, int2 per)
{
    float3  tmp = mx_cell_noise_float3_2((float2)(x+xoff, y+yoff), per);
    float2  off = (float2)(tmp.x, tmp.y);

    off -= 0.5f;
    off *= jitter;
    off += 0.5f;

    float2 cellpos = (float2)((float)(x), (float)(y)) + off;
    float2 diff = cellpos - p;
    if (metric == 2)
        return fabs(diff.x) + fabs(diff.y);       // Manhattan distance
    if (metric == 3)
        return max(fabs(diff.x), fabs(diff.y));   // Chebyshev distance
    // Either Euclidian or Distance^2
    return dot(diff, diff);
}

static float
mx_worley_distance3(float3 p, int x, int y, int z, int xoff, int yoff, int zoff,
                    float jitter, int metric, int3 per)
{
    float3  off = mx_cell_noise_float3_3((float3)(x+xoff, y+yoff, z+zoff), per);

    off -= 0.5f;
    off *= jitter;
    off += 0.5f;

    float3 cellpos = (float3)((float)(x), (float)(y), (float)(z)) + off;
    float3 diff = cellpos - p;
    if (metric == 2)
        return fabs(diff.x) + fabs(diff.y) + fabs(diff.z); // Manhattan distance
    if (metric == 3)
        return max(max(fabs(diff.x), fabs(diff.y)), fabs(diff.z)); // Chebyshev distance
    // Either Euclidian or Distance^2
    return dot(diff, diff);
}

static float
mx_worley_noise_float_2(float2 p, float jitter, int metric, int2 per)
{
    int X, Y;
    float2 localpos = (float2)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y));
    float sqdist = 1e6f;        // Some big number for jitter > 1 (not all GPUs may be IEEE)
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float dist = mx_worley_distance2(localpos, x, y, X, Y, jitter, metric, per);
            sqdist = min(sqdist, dist);
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float2
mx_worley_noise_float2_2(float2 p, float jitter, int metric, int2 per)
{
    int X, Y;
    float2 localpos = (float2)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y));
    float2 sqdist = (float2)(1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float dist = mx_worley_distance2(localpos, x, y, X, Y, jitter, metric, per);
            if (dist < sqdist.x)
            {
                sqdist.y = sqdist.x;
                sqdist.x = dist;
            }
            else if (dist < sqdist.y)
            {
                sqdist.y = dist;
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float3
mx_worley_noise_float3_2(float2 p, float jitter, int metric, int2 per)
{
    int X, Y;
    float2 localpos = (float2)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y));
    float3 sqdist = (float3)(1e6f, 1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float dist = mx_worley_distance2(localpos, x, y, X, Y, jitter, metric, per);
            if (dist < sqdist.x)
            {
                sqdist.z = sqdist.y;
                sqdist.y = sqdist.x;
                sqdist.x = dist;
            }
            else if (dist < sqdist.y)
            {
                sqdist.z = sqdist.y;
                sqdist.y = dist;
            }
            else if (dist < sqdist.z)
            {
                sqdist.z = dist;
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float
mx_worley_noise_float_3(float3 p, float jitter, int metric, int3 per)
{
    int X, Y, Z;
    float3 localpos = (float3)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y), mx_floorfrac(p.z, &Z));
    float sqdist = 1e6f;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                float dist = mx_worley_distance3(localpos, x, y, z, X, Y, Z, jitter, metric, per);
                sqdist = min(sqdist, dist);
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float2
mx_worley_noise_float2_3(float3 p, float jitter, int metric, int3 per)
{
    int X, Y, Z;
    float3 localpos = (float3)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y), mx_floorfrac(p.z, &Z));
    float2 sqdist = (float2)(1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                float dist = mx_worley_distance3(localpos, x, y, z, X, Y, Z, jitter, metric, per);
                if (dist < sqdist.x)
                {
                    sqdist.y = sqdist.x;
                    sqdist.x = dist;
                }
                else if (dist < sqdist.y)
                {
                    sqdist.y = dist;
                }
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float3
mx_worley_noise_float3_3(float3 p, float jitter, int metric, int3 per)
{
    int X, Y, Z;
    float3 localpos = (float3)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y), mx_floorfrac(p.z, &Z));
    float3 sqdist = (float3)(1e6f, 1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                float dist = mx_worley_distance3(localpos, x, y, z, X, Y, Z, jitter, metric, per);
                if (dist < sqdist.x)
                {
                    sqdist.z = sqdist.y;
                    sqdist.y = sqdist.x;
                    sqdist.x = dist;
                }
                else if (dist < sqdist.y)
                {
                    sqdist.z = sqdist.y;
                    sqdist.y = dist;
                }
                else if (dist < sqdist.z)
                {
                    sqdist.z = dist;
                }
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

#endif

#ifndef __ALLIGATOR_H__
#define __ALLIGATOR_H__

float alligator3(float3 pos, int3 period, int seed, float contrast)
{
    float3 ipos = floor(pos); // Integer Coordinates
    float3 fpos = pos - ipos; // Fractional Coordinates

    float densest = 0.0f;
    float secondDensest = 0.0f;
    for (int ix = -1; ix <= 1; ++ix)
    {
        for (int iy = -1; iy <= 1; ++iy)
        {        
            for (int iz = -1; iz <= 1; ++iz)
            {
                float3 offset = (float3)(ix, iy, iz);
                float3 cell = ipos + offset;
                
                // Wrap for tiling
                cell.x = mx_wrap_int(cell.x, period.x);
                cell.y = mx_wrap_int(cell.y, period.y);
                cell.z = mx_wrap_int(cell.z, period.z);
                cell += (float3)(seed);

                // VEX hash function... make sure to include random.h
                float3 center = VEXrand_3_3(cell.x, cell.y, cell.z) + offset;
                float dist = distance(fpos, center);

                if (dist < 1)
                {
                    float smoothDist = clamp(contrast - dist, 0.0f, 1.0f);
                    smoothDist = smoothDist * smoothDist * (3.0f - 2.0f * smoothDist);

                    // Another VEX hash function
                    float density = VEXrand_3_1(cell.x, cell.y, cell.z) * smoothDist;

                    if (densest < density)
                    {
                        secondDensest = densest;
                        densest = density;
                    }
                    else if (secondDensest < density)
                    {
                        secondDensest = density;
                    }
                }
            }
        }
    }
    float result = densest - secondDensest;
    if (result >= 0.2f)
    {
        result = fit(result, 0.2f, 1.0f, 0.5f, 1.0f);
    } else
    {
        result = fit(result, 0.0f, 0.2f, 0.0f, 0.5f);
    }
    
    return result;

}
float alligator2(float2 pos, int2 period, int seed, float contrast)
{
    float2 ipos = floor(pos); // Integer Coordinates
    float2 fpos = pos - ipos; // Fractional Coordinates

    float densest = 0.0f;
    float secondDensest = 0.0f;
    for (int ix = -1; ix <= 1; ++ix)
    {
        for (int iy = -1; iy <= 1; ++iy)
        {        
            float2 offset = (float2)(ix, iy);
            float2 cell = ipos + offset;
                
            // Wrap for tiling
            cell.x = mx_wrap_int(cell.x, period.x);
            cell.y = mx_wrap_int(cell.y, period.y);
            cell += (float2)(seed);

            // VEX hash function... make sure to include random.h
            float2 center = VEXrand_2_2(cell.x, cell.y) + offset;
            float dist = distance(fpos, center);

            if (dist < 1)
            {
                float smoothDist = clamp(contrast - dist, 0.0f, 1.0f);
                smoothDist = smoothDist * smoothDist * (3.0f - 2.0f * smoothDist);

                // Another VEX hash function
                float density = VEXrand_2_1(cell.x, cell.y) * smoothDist;

                if (densest < density)
                {
                    secondDensest = densest;
                    densest = density;
                }
                else if (secondDensest < density)
                {
                    secondDensest = density;
                }
            }
        }
    }
    float result = densest - secondDensest;
    if (result >= 0.35f)
    {
        result = fit(result, 0.35f, 1.0f, 0.5f, 1.0f);
    } else
    {
        result = fit(result, 0.0f, 0.35f, 0.0f, 0.5f);
    }
    
    return result;

}
#endif#else
/*
Noise Library.

This library is a modified version of the noise library found in
Open Shading Language:
github.com/imageworks/OpenShadingLanguage/blob/master/src/include/OSL/oslnoise.h

It contains the subset of noise types needed to implement the MaterialX
standard library. The modifications are mainly conversions from C++ to GLSL.
Produced results should be identical to the OSL noise functions.

Original copyright notice:
------------------------------------------------------------------------
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
------------------------------------------------------------------------
*/

#ifndef __MX_NOISE_INTERNAL_H__
#define __MX_NOISE_INTERNAL_H__

static int
mx_wrap_int(int x, int period)
{
    if (period <= 0)
        return x;
    int y = x - x / period * period;
    return y >= 0 ? y : y + period;
}

static float
mx_select(bool b, float t, float f)
{
    return b ? t : f;
}

static float
mx_negate_if(float val, bool b)
{
    return b ? -val : val;
}

static int
mx_floor(float x)
{
    return (int)(floor(x));
}

// return mx_floor as well as the fractional remainder
static float
mx_floorfrac(float x, int* i)
{
    *i = mx_floor(x);
    return x - (float)(*i);
}

static float
mx_bilerp1(float v0, float v1, float v2, float v3, float s, float t)
{
    float s1 = 1.0f - s;
    return (1.0f - t) * (v0*s1 + v1*s) + t * (v2*s1 + v3*s);
}
static float3
mx_bilerp3(float3 v0, float3 v1, float3 v2, float3 v3,
           float s, float t)
{
    float s1 = 1.0f - s;
    return (1.0f - t) * (v0*s1 + v1*s) + t * (v2*s1 + v3*s);
}
static float
mx_trilerp1(float v0, float v1, float v2, float v3,
            float v4, float v5, float v6, float v7,
            float s, float t, float r)
{
    float s1 = 1.0f - s;
    float t1 = 1.0f - t;
    float r1 = 1.0f - r;
    return (r1*(t1*(v0*s1 + v1*s) + t*(v2*s1 + v3*s)) +
            r*(t1*(v4*s1 + v5*s) + t*(v6*s1 + v7*s)));
}
static float3
mx_trilerp3(float3 v0, float3 v1, float3 v2, float3 v3,
            float3 v4, float3 v5, float3 v6, float3 v7,
            float s, float t, float r)
{
    float s1 = 1.0f - s;
    float t1 = 1.0f - t;
    float r1 = 1.0f - r;
    return (r1*(t1*(v0*s1 + v1*s) + t*(v2*s1 + v3*s)) +
            r*(t1*(v4*s1 + v5*s) + t*(v6*s1 + v7*s)));
}

// 2 and 3 dimensional gradient functions - perform a dot product against a
// randomly chosen vector. Note that the gradient vector is not normalized, but
// this only affects the overal "scale" of the result, so we simply account for
// the scale by multiplying in the corresponding "perlin" function.
static float
mx_gradient_float1_2(uint hash, float x, float y)
{
    // 8 possible directions (+-1,+-2) and (+-2,+-1)
    uint h = hash & 7u;
    float u = mx_select(h<4u, x, y);
    float v = 2.0 * mx_select(h<4u, y, x);
    // compute the dot product with (x,y).
    return mx_negate_if(u, (bool)(h&1u)) + mx_negate_if(v, (bool)(h&2u));
}
static float
mx_gradient_float1_3(uint hash, float x, float y, float z)
{
    // use vectors pointing to the edges of the cube
    uint h = hash & 15u;
    float u = mx_select(h<8u, x, y);
    float v = mx_select(h<4u, y, mx_select((h==12u)||(h==14u), x, z));
    return mx_negate_if(u, (bool)(h&1u)) + mx_negate_if(v, (bool)(h&2u));
}
static float3
mx_gradient_float3_2(uint3 hash, float x, float y)
{
    return (float3)(mx_gradient_float1_2(hash.x, x, y),
                    mx_gradient_float1_2(hash.y, x, y),
                    mx_gradient_float1_2(hash.z, x, y));
}
static float3
mx_gradient_float3_3(uint3 hash, float x, float y, float z)
{
    return (float3)(mx_gradient_float1_3(hash.x, x, y, z),
                    mx_gradient_float1_3(hash.y, x, y, z),
                    mx_gradient_float1_3(hash.z, x, y, z));
}
// Scaling factors to normalize the result of gradients above.
// These factors were experimentally calculated to be:
//    2D:   0.6616
//    3D:   0.9820
static float mx_gradient_scale2d_1(float v) { return 0.6616f * v; }
static float mx_gradient_scale3d_1(float v) { return 0.9820f * v; }
static float3 mx_gradient_scale2d_3(float3 v) { return 0.6616f * v; }
static float3 mx_gradient_scale3d_3(float3 v) { return 0.9820f * v; }

/// Bitwise circular rotation left by k bits (for 32 bit unsigned integers)
static uint
mx_rotl32(uint x, int k)
{
    return (x<<k) | (x>>(32-k));
}

static void
mx_bjmix(uint* a, uint* b, uint* c)
{
    *a -= *c; *a ^= mx_rotl32(*c, 4); *c += *b;
    *b -= *a; *b ^= mx_rotl32(*a, 6); *a += *c;
    *c -= *b; *c ^= mx_rotl32(*b, 8); *b += *a;
    *a -= *c; *a ^= mx_rotl32(*c,16); *c += *b;
    *b -= *a; *b ^= mx_rotl32(*a,19); *a += *c;
    *c -= *b; *c ^= mx_rotl32(*b, 4); *b += *a;
}

// Mix up and combine the bits of a, b, and c (doesn't change them, but
// returns a hash of those three original values).
static uint
mx_bjfinal(uint a, uint b, uint c)
{
    c ^= b; c -= mx_rotl32(b,14);
    a ^= c; a -= mx_rotl32(c,11);
    b ^= a; b -= mx_rotl32(a,25);
    c ^= b; c -= mx_rotl32(b,16);
    a ^= c; a -= mx_rotl32(c,4);
    b ^= a; b -= mx_rotl32(a,14);
    c ^= b; c -= mx_rotl32(b,24);
    return c;
}

// Convert a 32 bit integer into a floating point number in [0,1]
static float
mx_bits_to_01(uint bits)
{
    return (float)(bits) / (float)((uint)(0xffffffff));
}

static float
mx_fade(float t)
{
   return t * t * t * (t * (t * 6.0f- 15.0f) + 10.0f);
}

static uint
mx_hash_int1p(int x, int px)
{
    x = mx_wrap_int(x, px);
    
    uint len = 1u;
    uint seed = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    return mx_bjfinal(seed+(uint)(x), seed, seed);
}

static uint
mx_hash_int2p(int x, int y, int px, int py)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);

    uint len = 2u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    return mx_bjfinal(a, b, c);
}

static uint
mx_hash_int3p(int x, int y, int z, int px, int py, int pz)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);
    
    uint len = 3u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    c += (uint)(z);
    return mx_bjfinal(a, b, c);
}

static uint
mx_hash_int4p(int x, int y, int z, int xx, int px, int py, int pz, int pxx)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);
    xx = mx_wrap_int(xx, pxx);
    
    uint len = 4u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    c += (uint)(z);
    mx_bjmix(&a, &b, &c);
    a += (uint)(xx);
    return mx_bjfinal(a, b, c);
}

static uint
mx_hash_int4pb(int x, int y, int z, int xx, int yy,
               int px, int py, int pz, int pxx)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);
    xx = mx_wrap_int(xx, pxx);

    uint len = 5u;
    uint a, b, c;
    a = b = c = (uint)(0xdeadbeef) + (len << 2u) + 13u;
    a += (uint)(x);
    b += (uint)(y);
    c += (uint)(z);
    mx_bjmix(&a, &b, &c);
    a += (uint)(xx);
    b += (uint)(yy);
    return mx_bjfinal(a, b, c);
}

static uint3
mx_hash_float3_2p(int x, int y, int px, int py)
{
    uint h = mx_hash_int2p(x, y, px, py);
    // we only need the low-order bits to be random, so split out
    // the 32 bit result into 3 parts for each channel
    uint3 result;
    result.x = (h      ) & 0xFFu;
    result.y = (h >> 8 ) & 0xFFu;
    result.z = (h >> 16) & 0xFFu;
    return result;
}

static uint3
mx_hash_float3_3p(int x, int y, int z, int px, int py, int pz)
{
    x = mx_wrap_int(x, px);
    y = mx_wrap_int(y, py);
    z = mx_wrap_int(z, pz);

    uint h = mx_hash_int3p(x, y, z, px, py, pz);
    // we only need the low-order bits to be random, so split out
    // the 32 bit result into 3 parts for each channel
    uint3 result;
    result.x = (h      ) & 0xFFu;
    result.y = (h >> 8 ) & 0xFFu;
    result.z = (h >> 16) & 0xFFu;
    return result;
}

static float
mx_perlin_noise_float_2(float2 p, int2 per)
{
    int X, Y;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float result = mx_bilerp1(
        mx_gradient_float1_2(mx_hash_int2p(X  , Y  , per.x, per.y), fx    , fy    ),
        mx_gradient_float1_2(mx_hash_int2p(X+1, Y  , per.x, per.y), fx-1.0f, fy    ),
        mx_gradient_float1_2(mx_hash_int2p(X  , Y+1, per.x, per.y), fx    , fy-1.0f),
        mx_gradient_float1_2(mx_hash_int2p(X+1, Y+1, per.x, per.y), fx-1.0f, fy-1.0f),
        u, v);
    return mx_gradient_scale2d_1(result);
}

static float
mx_perlin_noise_float_3(float3 p, int3 per)
{
    int X, Y, Z;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float fz = mx_floorfrac(p.z, &Z);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float w = mx_fade(fz);
    float result = mx_trilerp1(
        mx_gradient_float1_3(mx_hash_int3p(X  , Y  , Z  , per.x, per.y, per.z),
                             fx    , fy    , fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y  , Z  , per.x, per.y, per.z),
                             fx-1.0f, fy    , fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X  , Y+1, Z  , per.x, per.y, per.z),
                             fx    , fy-1.0f, fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y+1, Z  , per.x, per.y, per.z),
                             fx-1.0f, fy-1.0f, fz    ),
        mx_gradient_float1_3(mx_hash_int3p(X  , Y  , Z+1, per.x, per.y, per.z),
                             fx    , fy    , fz-1.0f),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y  , Z+1, per.x, per.y, per.z),
                             fx-1.0f, fy    , fz-1.0f),
        mx_gradient_float1_3(mx_hash_int3p(X  , Y+1, Z+1, per.x, per.y, per.z),
                             fx    , fy-1.0f, fz-1.0f),
        mx_gradient_float1_3(mx_hash_int3p(X+1, Y+1, Z+1, per.x, per.y, per.z),
                             fx-1.0f, fy-1.0f, fz-1.0f),
        u, v, w);
    return mx_gradient_scale3d_1(result);
}

static float3
mx_perlin_noise_float3_2(float2 p, int2 per)
{
    int X, Y;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float3 result = mx_bilerp3(
        mx_gradient_float3_2(mx_hash_float3_2p(X  , Y  , per.x, per.y), fx    , fy    ),
        mx_gradient_float3_2(mx_hash_float3_2p(X+1, Y  , per.x, per.y), fx-1.0f, fy    ),
        mx_gradient_float3_2(mx_hash_float3_2p(X  , Y+1, per.x, per.y), fx    , fy-1.0f),
        mx_gradient_float3_2(mx_hash_float3_2p(X+1, Y+1, per.x, per.y), fx-1.0f, fy-1.0f),
        u, v);
    return mx_gradient_scale2d_3(result);
}

static float3
mx_perlin_noise_float3_3(float3 p, int3 per)
{
    int X, Y, Z;
    float fx = mx_floorfrac(p.x, &X);
    float fy = mx_floorfrac(p.y, &Y);
    float fz = mx_floorfrac(p.z, &Z);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float w = mx_fade(fz);
    float3 result = mx_trilerp3(
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y  , Z  , per.x, per.y, per.z),
                         fx    , fy    , fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y  , Z  , per.x, per.y, per.z),
                         fx-1.0f, fy    , fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y+1, Z  , per.x, per.y, per.z),
                         fx    , fy-1.0f, fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y+1, Z  , per.x, per.y, per.z),
                         fx-1.0f, fy-1.0f, fz    ),
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y  , Z+1, per.x, per.y, per.z),
                         fx    , fy    , fz-1.0f),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y  , Z+1, per.x, per.y, per.z),
                         fx-1.0f, fy    , fz-1.0f),
        mx_gradient_float3_3(mx_hash_float3_3p(X  , Y+1, Z+1, per.x, per.y, per.z),
                         fx    , fy-1.0f, fz-1.0f),
        mx_gradient_float3_3(mx_hash_float3_3p(X+1, Y+1, Z+1, per.x, per.y, per.z),
                         fx-1.0f, fy-1.0f, fz-1.0f),
        u, v, w);
    return mx_gradient_scale3d_3(result);
}

static float
mx_cell_noise_float_1(float p, int per)
{
    int ix = mx_floor(p);
    return mx_bits_to_01(mx_hash_int1p(ix, per));
}

static float
mx_cell_noise_float_2(float2 p, int2 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    return mx_bits_to_01(mx_hash_int2p(ix, iy, per.x, per.y));
}

static float
mx_cell_noise_float_3(float3 p, int3 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    return mx_bits_to_01(mx_hash_int3p(ix, iy, iz, per.x, per.y, per.z));
}

static float
mx_cell_noise_float_4(float4 p, int4 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    int iw = mx_floor(p.w);
    return mx_bits_to_01(mx_hash_int4p(ix, iy, iz, iw,
                                     per.x, per.y, per.z, per.w));
}

static float3
mx_cell_noise_float3_1(float p, int per)
{
    int ix = mx_floor(p);
    return (float3)(
            mx_bits_to_01(mx_hash_int2p(ix, 0, per, 0)),
            mx_bits_to_01(mx_hash_int2p(ix, 1, per, 0)),
            mx_bits_to_01(mx_hash_int2p(ix, 2, per, 0))
    );
}

static float3
mx_cell_noise_float3_2(float2 p, int2 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    return (float3)(
            mx_bits_to_01(mx_hash_int3p(ix, iy, 0, per.x, per.y, 0)),
            mx_bits_to_01(mx_hash_int3p(ix, iy, 1, per.x, per.y, 0)),
            mx_bits_to_01(mx_hash_int3p(ix, iy, 2, per.x, per.y, 0))
    );
}

static float3
mx_cell_noise_float3_3(float3 p, int3 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    return (float3)(
            mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 0, per.x, per.y, per.z, 0)),
            mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 1, per.x, per.y, per.z, 0)),
            mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 2, per.x, per.y, per.z, 0))
    );
}

static float3
mx_cell_noise_float3_4(float4 p, int4 per)
{
    int ix = mx_floor(p.x);
    int iy = mx_floor(p.y);
    int iz = mx_floor(p.z);
    int iw = mx_floor(p.w);
    return (float3)(
            mx_bits_to_01(mx_hash_int4pb(ix, iy, iz, iw, 0,
                                         per.x, per.y, per.z, per.w)),
            mx_bits_to_01(mx_hash_int4pb(ix, iy, iz, iw, 1,
                                         per.x, per.y, per.z, per.w)),
            mx_bits_to_01(mx_hash_int4pb(ix, iy, iz, iw, 2,
                                         per.x, per.y, per.z, per.w))
    );
}

static float
mx_fractal_noise_float(float3 p, int octaves, float lacunarity, float diminish,
                       int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.
    
    float result = 0.0f;
    float amplitude = 1.0;
    for (int i = 0;  i < octaves; ++i)
    {
        result += amplitude * mx_perlin_noise_float_3(p, per);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}

static float3
mx_fractal_noise_float3(float3 p, int octaves, float lacunarity, float diminish,
                        int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.

    float3 result = (float3)(0.0f);
    float amplitude = 1.0;
    for (int i = 0;  i < octaves; ++i)
    {
        result += amplitude * mx_perlin_noise_float3_3(p, per);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}

static float2
mx_fractal_noise_float2(float3 p, int octaves, float lacunarity, float diminish,
                        int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.
    // Offsetting positions for the second component results in the exact same
    // value for both if offset is an integral vector multiple of period.

    return (float2)(mx_fractal_noise_float(p, octaves, lacunarity,
                                           diminish, per),
                    mx_fractal_noise_float(p+(float3)(19, 193, 17), octaves, lacunarity,
                                           diminish, per));
}

static float4
mx_fractal_noise_float4(float3 p, int octaves, float lacunarity, float diminish,
                        int3 per)
{
    // If period is positive along some axis, lacunarity has to be an integer to
    // preserve tileability.

    float3 c = mx_fractal_noise_float3(p, octaves, lacunarity, diminish, per);
    float f = mx_fractal_noise_float(p+(float3)(19, 193, 17), octaves, lacunarity,
                                     diminish, per);
    return (float4)(c, f);
}

static float
mx_worley_distance2(float2 p, int x, int y, int xoff, int yoff, float jitter,
                    int metric, int2 per)
{
    float3  tmp = mx_cell_noise_float3_2((float2)(x+xoff, y+yoff), per);
    float2  off = (float2)(tmp.x, tmp.y);

    off -= 0.5f;
    off *= jitter;
    off += 0.5f;

    float2 cellpos = (float2)((float)(x), (float)(y)) + off;
    float2 diff = cellpos - p;
    if (metric == 2)
        return fabs(diff.x) + fabs(diff.y);       // Manhattan distance
    if (metric == 3)
        return max(fabs(diff.x), fabs(diff.y));   // Chebyshev distance
    // Either Euclidian or Distance^2
    return dot(diff, diff);
}

static float
mx_worley_distance3(float3 p, int x, int y, int z, int xoff, int yoff, int zoff,
                    float jitter, int metric, int3 per)
{
    float3  off = mx_cell_noise_float3_3((float3)(x+xoff, y+yoff, z+zoff), per);

    off -= 0.5f;
    off *= jitter;
    off += 0.5f;

    float3 cellpos = (float3)((float)(x), (float)(y), (float)(z)) + off;
    float3 diff = cellpos - p;
    if (metric == 2)
        return fabs(diff.x) + fabs(diff.y) + fabs(diff.z); // Manhattan distance
    if (metric == 3)
        return max(max(fabs(diff.x), fabs(diff.y)), fabs(diff.z)); // Chebyshev distance
    // Either Euclidian or Distance^2
    return dot(diff, diff);
}

static float
mx_worley_noise_float_2(float2 p, float jitter, int metric, int2 per)
{
    int X, Y;
    float2 localpos = (float2)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y));
    float sqdist = 1e6f;        // Some big number for jitter > 1 (not all GPUs may be IEEE)
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float dist = mx_worley_distance2(localpos, x, y, X, Y, jitter, metric, per);
            sqdist = min(sqdist, dist);
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float2
mx_worley_noise_float2_2(float2 p, float jitter, int metric, int2 per)
{
    int X, Y;
    float2 localpos = (float2)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y));
    float2 sqdist = (float2)(1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float dist = mx_worley_distance2(localpos, x, y, X, Y, jitter, metric, per);
            if (dist < sqdist.x)
            {
                sqdist.y = sqdist.x;
                sqdist.x = dist;
            }
            else if (dist < sqdist.y)
            {
                sqdist.y = dist;
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float3
mx_worley_noise_float3_2(float2 p, float jitter, int metric, int2 per)
{
    int X, Y;
    float2 localpos = (float2)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y));
    float3 sqdist = (float3)(1e6f, 1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float dist = mx_worley_distance2(localpos, x, y, X, Y, jitter, metric, per);
            if (dist < sqdist.x)
            {
                sqdist.z = sqdist.y;
                sqdist.y = sqdist.x;
                sqdist.x = dist;
            }
            else if (dist < sqdist.y)
            {
                sqdist.z = sqdist.y;
                sqdist.y = dist;
            }
            else if (dist < sqdist.z)
            {
                sqdist.z = dist;
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float
mx_worley_noise_float_3(float3 p, float jitter, int metric, int3 per)
{
    int X, Y, Z;
    float3 localpos = (float3)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y), mx_floorfrac(p.z, &Z));
    float sqdist = 1e6f;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                float dist = mx_worley_distance3(localpos, x, y, z, X, Y, Z, jitter, metric, per);
                sqdist = min(sqdist, dist);
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float2
mx_worley_noise_float2_3(float3 p, float jitter, int metric, int3 per)
{
    int X, Y, Z;
    float3 localpos = (float3)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y), mx_floorfrac(p.z, &Z));
    float2 sqdist = (float2)(1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                float dist = mx_worley_distance3(localpos, x, y, z, X, Y, Z, jitter, metric, per);
                if (dist < sqdist.x)
                {
                    sqdist.y = sqdist.x;
                    sqdist.x = dist;
                }
                else if (dist < sqdist.y)
                {
                    sqdist.y = dist;
                }
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

static float3
mx_worley_noise_float3_3(float3 p, float jitter, int metric, int3 per)
{
    int X, Y, Z;
    float3 localpos = (float3)(mx_floorfrac(p.x, &X), mx_floorfrac(p.y, &Y), mx_floorfrac(p.z, &Z));
    float3 sqdist = (float3)(1e6f, 1e6f, 1e6f);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                float dist = mx_worley_distance3(localpos, x, y, z, X, Y, Z, jitter, metric, per);
                if (dist < sqdist.x)
                {
                    sqdist.z = sqdist.y;
                    sqdist.y = sqdist.x;
                    sqdist.x = dist;
                }
                else if (dist < sqdist.y)
                {
                    sqdist.z = sqdist.y;
                    sqdist.y = dist;
                }
                else if (dist < sqdist.z)
                {
                    sqdist.z = dist;
                }
            }
        }
    }
    if (metric == 0)
        sqdist = sqrt(sqdist);
    return sqdist;
}

#endif

#ifndef __mtlx_NOISE_H__
#define __mtlx_NOISE_H__

// #include "mtlx_noise_internal.h"

/// mtlx_noise2d_X: equivalent to MaterialX noise2d node, produces X channels.
/// Perlin noise from 2D space.
static float
mtlx_noise2d_1(float amplitude, float pivot, float2 texcoord, int2 period)
{
    float value = mx_perlin_noise_float_2(texcoord, period);
    return value * amplitude + pivot;
}
static float2
mtlx_noise2d_2(float2 amplitude, float2 pivot, float2 texcoord, int2 period)
{
    float3 value = mx_perlin_noise_float3_2(texcoord, period);
    return value.xy * amplitude + pivot;
}
static float3
mtlx_noise2d_3(float3 amplitude, float3 pivot, float2 texcoord, int2 period)
{
    float3 value = mx_perlin_noise_float3_2(texcoord, period);
    return value * amplitude + pivot;
}
static float4
mtlx_noise2d_4(float4 amplitude, float4 pivot, float2 texcoord, int2 period)
{
    float3 xyz = mx_perlin_noise_float3_2(texcoord, period);
    float w = mx_perlin_noise_float_2(texcoord + (float2)(19.0, 73.0), period);
    return (float4)(xyz, w) * amplitude + pivot;
}

/// mtlx_noise3d_X: equivalent to MaterialX noise3d node, produces X channels.
/// Perlin noise from 3D space.
static float
mtlx_noise3d_1(float amplitude, float pivot, float3 position, int3 period)
{
    float value = mx_perlin_noise_float_3(position, period);
    return value * amplitude + pivot;
}
static float2
mtlx_noise3d_2(float2 amplitude, float2 pivot, float3 position, int3 period)
{
    float3 value = mx_perlin_noise_float3_3(position, period);
    return value.xy * amplitude + pivot;
}
static float3
mtlx_noise3d_3(float3 amplitude, float3 pivot, float3 position, int3 period)
{
    float3 value = mx_perlin_noise_float3_3(position, period);
    return value * amplitude + pivot;
}
static float4
mtlx_noise3d_4(float4 amplitude, float4 pivot, float3 position, int3 period)
{
    float3 xyz = mx_perlin_noise_float3_3(position, period);
    float w = mx_perlin_noise_float_3(position + (float3)(19.0, 73.0, 29.0),
                                      period);
    return (float4)(xyz, w) * amplitude + pivot;
}

/// mtlx_fractal3d_X: equivalent to MaterialX fractal3d node, produces X
/// channels. Combination of several octaves of Perlin noise from 3D space.
static float
mtlx_fractal3d_1(float amplitude, int octaves, float lacunarity, float diminish,
                 float3 position, int3 period)
{
    float value = mx_fractal_noise_float(position, octaves, lacunarity,
                                         diminish, period);
    return value * amplitude;
}
static float2
mtlx_fractal3d_2(float2 amplitude, int octaves, float lacunarity,
                 float diminish, float3 position, int3 period)
{
    float2 value = mx_fractal_noise_float2(position, octaves, lacunarity,
                                           diminish, period);
    return value * amplitude;
}
static float3
mtlx_fractal3d_3(float3 amplitude, int octaves, float lacunarity,
                 float diminish, float3 position, int3 period)
{
    float3 value = mx_fractal_noise_float3(position, octaves, lacunarity,
                                           diminish, period);
    return value * amplitude;
}
static float4
mtlx_fractal3d_4(float4 amplitude, int octaves, float lacunarity,
                 float diminish, float3 position, int3 period)
{
    float4 value = mx_fractal_noise_float4(position, octaves, lacunarity,
                                           diminish, period);
    return value * amplitude;
}

/// mtlx_cellnoise2d_1: equivalent to MaterialX cellnoise2d node, produces 1
/// channel from 2D position.
static float
mtlx_cellnoise2d_1(float2 texcoord, int2 period)
{
    return mx_cell_noise_float_2(texcoord, period);
}

/// mtlx_cellnoise3d_1: equivalent to MaterialX cellnoise3d node, produces 1
/// channel from 3D position.
static float
mtlx_cellnoise3d_1(float3 position, int3 period)
{
    return mx_cell_noise_float_3(position, period);
}

/// mtlx_worleynoise2d_X: equivalent to MaterialX worleynoise2d node, produces X
/// channels. Worley noise from 2D position.
static float
mtlx_worleynoise2d_1(float2 texcoord, float jitter, int2 period)
{
    return mx_worley_noise_float_2(texcoord, jitter, 0, period);
}
static float2
mtlx_worleynoise2d_2(float2 texcoord, float jitter, int2 period)
{
    return mx_worley_noise_float2_2(texcoord, jitter, 0, period);
}
static float3
mtlx_worleynoise2d_3(float2 texcoord, float jitter, int2 period)
{
    return mx_worley_noise_float3_2(texcoord, jitter, 0, period);
}

/// mtlx_worleynoise3d_X: equivalent to MaterialX worleynoise3d node, produces X
/// channels. Worley noise from 3D position.
static float
mtlx_worleynoise3d_1(float3 position, float jitter, int3 period)
{
    return mx_worley_noise_float_3(position, jitter, 0, period);
}
static float2
mtlx_worleynoise3d_2(float3 position, float jitter, int3 period)
{
    return mx_worley_noise_float2_3(position, jitter, 0, period);
}
static float3
mtlx_worleynoise3d_3(float3 position, float jitter, int3 period)
{
    return mx_worley_noise_float3_3(position, jitter, 0, period);
}

#endif

#endif

#define CREATE_APPLY_CONTRAST_FUNC(TYPE) \
static TYPE \
_apply_contrast_ ## TYPE(TYPE x, float contrast_exponent) \
{ \
    return select(0.5f * pow(1.0f + 2.0f * x, contrast_exponent) - 0.5f, \
                  0.5f - 0.5f * pow(1.0f - 2.0f * x, contrast_exponent), \
                  x > 0.0f); \
}
CREATE_APPLY_CONTRAST_FUNC(float)
CREATE_APPLY_CONTRAST_FUNC(float2)
CREATE_APPLY_CONTRAST_FUNC(float3)
CREATE_APPLY_CONTRAST_FUNC(float4)

#if NOISETYPE == NOISE_PERLIN
static float 
_fbm_noisewrap_perlin2_1(float2 pos, int2 period, float c)
{
    float v = mx_perlin_noise_float_2(pos, period) * 0.5f;
    return _apply_contrast_float(v, c);
}
static float2
_fbm_noisewrap_perlin2_2(float2 pos, int2 period, float c)
{
    float3 v = mx_perlin_noise_float3_2(pos, period) * 0.5f;
    return _apply_contrast_float2(v.xy, c);
}
static float3
_fbm_noisewrap_perlin2_3(float2 pos, int2 period, float c)
{
    float3 v = mx_perlin_noise_float3_2(pos, period) * 0.5f;
    return _apply_contrast_float3(v, c);
}
static float4
_fbm_noisewrap_perlin2_4(float2 pos, int2 period, float c)
{
    float3 xyz = mx_perlin_noise_float3_2(pos, period);
    float w = mx_perlin_noise_float_2(pos + (float2)(19.0, 73.0), period);
    float4 v = (float4)(xyz, w) * 0.5f;
    return _apply_contrast_float4(v, c);
}

static float
_fbm_noisewrap_perlin3_1(float3 pos, int3 period, float c)
{
    float v = mx_perlin_noise_float_3(pos, period) * 0.5f;
    return _apply_contrast_float(v, c);
}
static float2
_fbm_noisewrap_perlin3_2(float3 pos, int3 period, float c)
{
    float3 v = mx_perlin_noise_float3_3(pos, period) * 0.5f;
    return _apply_contrast_float2(v.xy, c);
}
static float3
_fbm_noisewrap_perlin3_3(float3 pos, int3 period, float c)
{
    float3 v = mx_perlin_noise_float3_3(pos, period) * 0.5f;
    return _apply_contrast_float3(v, c);
}
static float4
_fbm_noisewrap_perlin3_4(float3 pos, int3 period, float c)
{
    float3 xyz = mx_perlin_noise_float3_3(pos, period);
    float w = mx_perlin_noise_float_3(pos + (float3)(19.0, 73.0, 123.0), period);
    float4 v = (float4)(xyz, w) * 0.5f;
    return _apply_contrast_float4(v, c);
}
#endif

#if NOISETYPE == NOISE_WORLEYFA

// Euclidean norm, max value is sqrt(DIM).
#if WORLEY_METRIC == 0
#define WORLEY_NORMALIZER_2 0.70710678118f
#define WORLEY_NORMALIZER_3 0.57735026919f
// Manhattan norm, max value is DIM.
#elif WORLEY_METRIC == 2
#define WORLEY_NORMALIZER_2 0.5f
#define WORLEY_NORMALIZER_3 0.33333333333f
// Chebyshev norm, max value is 1.
#elif WORLEY_METRIC == 3
#define WORLEY_NORMALIZER_2 1.0f
#define WORLEY_NORMALIZER_3 1.0f
#endif

static float
_fbm_noisewrap_worleyA2_1(float2 pos, int2 period, float c, float jitter)
{
    float v = mx_worley_noise_float_2(pos, jitter, WORLEY_METRIC, period)
            * WORLEY_NORMALIZER_2 - 0.5f;
    return _apply_contrast_float(v, c);
}

static float2
_fbm_noisewrap_worleyA2_2(float2 pos, int2 period, float c, float jitter)
{
    float2 v;
    v.x = mx_worley_noise_float_2(pos, jitter, WORLEY_METRIC, period);
    v.y = mx_worley_noise_float_2(pos + (float2)(173.123, 215.123), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_2;
    v -= 0.5f;
    return _apply_contrast_float2(v, c);
}

static float3
_fbm_noisewrap_worleyA2_3(float2 pos, int2 period, float c, float jitter)
{
    float3 v;
    v.x = mx_worley_noise_float_2(pos, jitter, WORLEY_METRIC, period);
    v.y = mx_worley_noise_float_2(pos + (float2)(173.123, 215.123), jitter, WORLEY_METRIC, period);
    v.z = mx_worley_noise_float_2(pos + (float2)(253.123, -173.123), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_2;
    v -= 0.5f;
    return _apply_contrast_float3(v, c);
}

static float4
_fbm_noisewrap_worleyA2_4(float2 pos, int2 period, float c, float jitter)
{
    float4 v;
    v.x = mx_worley_noise_float_2(pos, jitter, WORLEY_METRIC, period);
    v.y = mx_worley_noise_float_2(pos + (float2)(173.123, 215.123), jitter, WORLEY_METRIC, period);
    v.z = mx_worley_noise_float_2(pos + (float2)(253.123, -173.123), jitter, WORLEY_METRIC, period);
    v.w = mx_worley_noise_float_2(pos + (float2)(-253.123, -57.123), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_2;
    v -= 0.5f;
    return _apply_contrast_float4(v, c);
}


static float
_fbm_noisewrap_worleyA3_1(float3 pos, int3 period, float c, float jitter)
{
    float v = mx_worley_noise_float_3(pos, jitter, WORLEY_METRIC, period)
            * WORLEY_NORMALIZER_3 - 0.5f;
    return _apply_contrast_float(v, c);
}

static float2
_fbm_noisewrap_worleyA3_2(float3 pos, int3 period, float c, float jitter)
{
    float2 v;
    v.x = mx_worley_noise_float_3(pos, jitter, WORLEY_METRIC, period);
    v.y = mx_worley_noise_float_3(pos + (float3)(173.123, 215.123, 127), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_3;
    v -= 0.5f;
    return _apply_contrast_float2(v, c);
}

static float3
_fbm_noisewrap_worleyA3_3(float3 pos, int3 period, float c, float jitter)
{
    float3 v;
    v.x = mx_worley_noise_float_3(pos, jitter, WORLEY_METRIC, period);
    v.y = mx_worley_noise_float_3(pos + (float3)(173.123, 215.123, 127), jitter, WORLEY_METRIC, period);
    v.z = mx_worley_noise_float_3(pos + (float3)(253.123, -173.123, 127), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_3;
    v -= 0.5f;
    return _apply_contrast_float3(v, c);
}

static float4
_fbm_noisewrap_worleyA3_4(float3 pos, int3 period, float c, float jitter)
{
    float4 v;
    v.x = mx_worley_noise_float_3(pos, jitter, WORLEY_METRIC, period);
    v.y = mx_worley_noise_float_3(pos + (float3)(173.123, 215.123, 127), jitter, WORLEY_METRIC, period);
    v.z = mx_worley_noise_float_3(pos + (float3)(253.123, -173.123, 127), jitter, WORLEY_METRIC, period);
    v.w = mx_worley_noise_float_3(pos + (float3)(-253.123, -57.123, 127), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_3;
    v -= 0.5f;
    return _apply_contrast_float4(v, c);
}
#endif

#if NOISETYPE == NOISE_WORLEYFB

// Euclidean norm, max value is sqrt(DIM+8)/2.
#if WORLEY_METRIC == 0
#define WORLEY_NORMALIZER_2 0.63245553203f
#define WORLEY_NORMALIZER_3 0.60302268915f
// Manhattan norm, max value is 1+DIM/2.
#elif WORLEY_METRIC == 2
#define WORLEY_NORMALIZER_2 0.5f
#define WORLEY_NORMALIZER_3 0.4f
// Chebyshev norm, max value is 1.5.
#elif WORLEY_METRIC == 3
#define WORLEY_NORMALIZER_2 0.66666666666f
#define WORLEY_NORMALIZER_3 0.66666666666f
#endif

static float
_fbm_worley_F2F1_2(float2 pos, float jitter, int metric, int2 period)
{
    float2 f1f2 = mx_worley_noise_float2_2(pos, jitter, metric, period);
    return f1f2.y - f1f2.x;
}

static float
_fbm_noisewrap_worleyB2_1(float2 pos, int2 period, float c, float jitter)
{
    float v = _fbm_worley_F2F1_2(pos, jitter, WORLEY_METRIC, period)
            * WORLEY_NORMALIZER_2 - 0.5f;
    return _apply_contrast_float(v, c);
}

static float2
_fbm_noisewrap_worleyB2_2(float2 pos, int2 period, float c, float jitter)
{
    float2 v;
    v.x = _fbm_worley_F2F1_2(pos, jitter, WORLEY_METRIC, period);
    v.y = _fbm_worley_F2F1_2(pos + (float2)(173.123, 215.123), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_2;
    v -= 0.5f;
    return _apply_contrast_float2(v, c);
}

static float3
_fbm_noisewrap_worleyB2_3(float2 pos, int2 period, float c, float jitter)
{
    float3 v;
    v.x = _fbm_worley_F2F1_2(pos, jitter, WORLEY_METRIC, period);
    v.y = _fbm_worley_F2F1_2(pos + (float2)(173.123, 215.123), jitter, WORLEY_METRIC, period);
    v.z = _fbm_worley_F2F1_2(pos + (float2)(253.123, -173.123), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_2;
    v -= 0.5f;
    return _apply_contrast_float3(v, c);
}

static float4
_fbm_noisewrap_worleyB2_4(float2 pos, int2 period, float c, float jitter)
{
    float4 v;
    v.x = _fbm_worley_F2F1_2(pos, jitter, WORLEY_METRIC, period);
    v.y = _fbm_worley_F2F1_2(pos + (float2)(173.123, 215.123), jitter, WORLEY_METRIC, period);
    v.z = _fbm_worley_F2F1_2(pos + (float2)(253.123, -173.123), jitter, WORLEY_METRIC, period);
    v.w = _fbm_worley_F2F1_2(pos + (float2)(-253.123, -57.123), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_2;
    v -= 0.5f;
    return _apply_contrast_float4(v, c);
}

static float
_fbm_worley_F2F1_3(float3 pos, float jitter, int metric, int3 period)
{
    float2 f1f2 = mx_worley_noise_float2_3(pos, jitter, metric, period);
    return f1f2.y - f1f2.x;
}

static float
_fbm_noisewrap_worleyB3_1(float3 pos, int3 period, float c, float jitter)
{
    float v = _fbm_worley_F2F1_3(pos, jitter, WORLEY_METRIC, period)
            * WORLEY_NORMALIZER_3 - 0.5f;
    return _apply_contrast_float(v, c);
}

static float2
_fbm_noisewrap_worleyB3_2(float3 pos, int3 period, float c, float jitter)
{
    float2 v;
    v.x = _fbm_worley_F2F1_3(pos, jitter, WORLEY_METRIC, period);
    v.y = _fbm_worley_F2F1_3(pos + (float3)(173.123, 215.123, 127), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_3;
    v -= 0.5f;
    return _apply_contrast_float2(v, c);
}

static float3
_fbm_noisewrap_worleyB3_3(float3 pos, int3 period, float c, float jitter)
{
    float3 v;
    v.x = _fbm_worley_F2F1_3(pos, jitter, WORLEY_METRIC, period);
    v.y = _fbm_worley_F2F1_3(pos + (float3)(173.123, 215.123, 127), jitter, WORLEY_METRIC, period);
    v.z = _fbm_worley_F2F1_3(pos + (float3)(253.123, -173.123, 127), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_3;
    v -= 0.5f;
    return _apply_contrast_float3(v, c);
}

static float4
_fbm_noisewrap_worleyB3_4(float3 pos, int3 period, float c, float jitter)
{
    float4 v;
    v.x = _fbm_worley_F2F1_3(pos, jitter, WORLEY_METRIC, period);
    v.y = _fbm_worley_F2F1_3(pos + (float3)(173.123, 215.123, 127), jitter, WORLEY_METRIC, period);
    v.z = _fbm_worley_F2F1_3(pos + (float3)(253.123, -173.123, 127), jitter, WORLEY_METRIC, period);
    v.w = _fbm_worley_F2F1_3(pos + (float3)(-253.123, -57.123, 127), jitter, WORLEY_METRIC, period);
    v *= WORLEY_NORMALIZER_3;
    v -= 0.5f;
    return _apply_contrast_float4(v, c);
}
#endif

#if NOISETYPE == NOISE_TORUS
static float
_fbm_noisewrap_gxnoise4_1(float4 pos, int4 period, float c)
{
    float v = VEXgxnoise_4_1(pos) - 0.5f;
    return _apply_contrast_float(v, c);
}
static float2
_fbm_noisewrap_gxnoise4_2(float4 pos, int4 period, float c)
{
    float2 v = VEXgxnoise_4_2(pos) - 0.5f;
    return _apply_contrast_float2(v, c);
}
static float3
_fbm_noisewrap_gxnoise4_3(float4 pos, int4 period, float c)
{
    float3 v = VEXgxnoise_4_3(pos) - 0.5f;
    return _apply_contrast_float3(v, c);
}
static float4
_fbm_noisewrap_gxnoise4_4(float4 pos, int4 period, float c)
{
    float4 v = VEXgxnoise_4_4(pos) - 0.5f;
    return _apply_contrast_float4(v, c);
}
#endif

#if NOISETYPE == NOISE_WHITE
static float
_fbm_noisewrap_cellnoise2_1(float2 pos, int2 period, float c)
{
    float v = mx_cell_noise_float_2(pos, period) - 0.5f;
    return _apply_contrast_float(v, c);
}
static float2
_fbm_noisewrap_cellnoise2_2(float2 pos, int2 period, float c)
{
    float3 v = mx_cell_noise_float3_2(pos, period) - 0.5f;
    return _apply_contrast_float2(v.xy, c);
}
static float3
_fbm_noisewrap_cellnoise2_3(float2 pos, int2 period, float c)
{
    float3 v = mx_cell_noise_float3_2(pos, period) - 0.5f;
    return _apply_contrast_float3(v, c);
}
static float4
_fbm_noisewrap_cellnoise2_4(float2 pos, int2 period, float c)
{
    int ix = mx_floor(pos.x);
    int iy = mx_floor(pos.y);
    float4 v = (float4)(
        mx_bits_to_01(mx_hash_int3p(ix, iy, 0, period.x, period.y, 0)),
        mx_bits_to_01(mx_hash_int3p(ix, iy, 1, period.x, period.y, 0)),
        mx_bits_to_01(mx_hash_int3p(ix, iy, 2, period.x, period.y, 0)),
        mx_bits_to_01(mx_hash_int3p(ix, iy, 3, period.x, period.y, 0)));
   v -= 0.5f;
   return _apply_contrast_float4(v, c);
}
static float
_fbm_noisewrap_cellnoise3_1(float3 pos, int3 period, float c)
{
    float v = mx_cell_noise_float_3(pos, period) - 0.5f;
    return _apply_contrast_float(v, c);
}
static float2
_fbm_noisewrap_cellnoise3_2(float3 pos, int3 period, float c)
{
    float3 v = mx_cell_noise_float3_3(pos, period) - 0.5f;
    return _apply_contrast_float2(v.xy, c);
}
static float3
_fbm_noisewrap_cellnoise3_3(float3 pos, int3 period, float c)
{
    float3 v = mx_cell_noise_float3_3(pos, period) - 0.5f;
    return _apply_contrast_float3(v, c);
}
static float4
_fbm_noisewrap_cellnoise3_4(float3 pos, int3 period, float c)
{
    int ix = mx_floor(pos.x);
    int iy = mx_floor(pos.y);
    int iz = mx_floor(pos.z);
    float4 v = (float4)(
        mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 0, period.x, period.y, period.z, 0)),
        mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 1, period.x, period.y, period.z, 0)),
        mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 2, period.x, period.y, period.z, 0)),
        mx_bits_to_01(mx_hash_int4p(ix, iy, iz, 3, period.x, period.y, period.z, 0)));
   v -= 0.5f;
   return _apply_contrast_float4(v, c);
}
#endif
#if NOISETYPE == NOISE_ALLIGATOR
static float
_fbm_noisewrap_alligator2_1(float2 pos, int2 period, float c)
{
    float v = alligator2(pos, period, 1, 1.0f) - 0.5f;
    return _apply_contrast_float(v, c);
}
static float2
_fbm_noisewrap_alligator2_2(float2 pos, int2 period, float c)
{
    float2 v;
    v.x = alligator2(pos, period, 1, 1.0f) - 0.5f;
    v.y = alligator2(pos, period, 2, 1.0f) - 0.5f;
    return _apply_contrast_float2(v, c);
}
static float3 
_fbm_noisewrap_alligator2_3(float2 pos, int2 period, float c)
{
    float3 v;
    v.x = alligator2(pos, period, 1, 1.0f) - 0.5f;
    v.y = alligator2(pos, period, 2, 1.0f) - 0.5f;
    v.z = alligator2(pos, period, 3, 1.0f) - 0.5f;
    return _apply_contrast_float3(v, c);
}
static float4 
_fbm_noisewrap_alligator2_4(float2 pos, int2 period, float c)
{
    float4 v;
    v.x = alligator2(pos, period, 1, 1.0f);
    v.y = alligator2(pos, period, 2, 1.0f);
    v.z = alligator2(pos, period, 3, 1.0f);
    v.w = alligator2(pos, period, 4, 1.0f);
    v -= 0.5f;
    return _apply_contrast_float4(v, c);
}

    
static float
_fbm_noisewrap_alligator3_1(float3 pos, int3 period, float c)
{
    float v = alligator3(pos, period, 1, 1.0f) - 0.5f;
    return _apply_contrast_float(v, c);
}
static float2
_fbm_noisewrap_alligator3_2(float3 pos, int3 period, float c)
{ 
    float2 v;
    v.x = alligator3(pos, period, 1, 1.0f) - 0.5f;
    v.y = alligator3(pos, period, 2, 1.0f) - 0.5f;
    return _apply_contrast_float2(v, c);
}
static float3
_fbm_noisewrap_alligator3_3(float3 pos, int3 period, float c)
{
    float3 v;
    v.x = alligator3(pos, period, 1, 1.0f) - 0.5f;
    v.y = alligator3(pos, period, 2, 1.0f) - 0.5f;
    v.z = alligator3(pos, period, 3, 1.0f) - 0.5f;
    return _apply_contrast_float3(v, c);
}
static float4
_fbm_noisewrap_alligator3_4(float3 pos, int3 period, float c)
{
    float4 v;
    v.x = alligator3(pos, period, 1, 1.0f);
    v.y = alligator3(pos, period, 2, 1.0f);
    v.z = alligator3(pos, period, 3, 1.0f);
    v.w = alligator3(pos, period, 4, 1.0f);
    v -= 0.5f;
    return _apply_contrast_float4(v, c);
}
#endif



#if NOISETYPE == NOISE_WORLEYFA || NOISETYPE == NOISE_WORLEYFB
#define CALLFUNC(NOISENAME, PDIM, RDIM) NOISENAME##PDIM##_##RDIM(p, period, contrast, jitter);
#else
#define CALLFUNC(NOISENAME, PDIM, RDIM) NOISENAME##PDIM##_##RDIM(p, period, contrast);
#endif


#define FBM_BOX(NAME, NOISENAME, RTYPE, RDIM, PTYPE, PDIM) \
static RTYPE \
FBM##NAME##_##PDIM##_##RDIM(PTYPE p, int##PDIM period, float octaves, float roughness, float lacunarity, float contrast, float jitter) \
{ \
    float weight = 1; \
    float gain = roughness * min(lacunarity, 1.0F); \
    \
    RTYPE base = 0; \
    base = CALLFUNC(NOISENAME, PDIM, RDIM); \
    float oct = 0; \
    \
    if (oct >= octaves) return base; \
    \
    do \
    { \
        weight *= gain; \
        oct += 1; \
        if (oct >= octaves) \
        { \
            weight *= 1 - (oct - octaves); \
        } \
        p *= lacunarity; \
        period *= (int)lacunarity; \
        p += 1;         /* Shift */ \
        RTYPE adjust = 0; \
        adjust = CALLFUNC(NOISENAME, PDIM, RDIM); \
        base += adjust * weight; \
    } while (oct < octaves); \
    \
    return base; \
} \
/**/
 

#define BUILD_ALL_FBM(BASENAME, PDIM) \
FBM_BOX(BASENAME, _fbm_noisewrap_##BASENAME, float, 1, float##PDIM, PDIM) \
FBM_BOX(BASENAME, _fbm_noisewrap_##BASENAME, float2, 2, float##PDIM, PDIM) \
FBM_BOX(BASENAME, _fbm_noisewrap_##BASENAME, float3, 3, float##PDIM, PDIM) \
FBM_BOX(BASENAME, _fbm_noisewrap_##BASENAME, float4, 4, float##PDIM, PDIM) \

#if NOISETYPE == NOISE_PERLIN
BUILD_ALL_FBM(perlin, 2)
BUILD_ALL_FBM(perlin, 3)
#endif
#if NOISETYPE == NOISE_WORLEYFA
BUILD_ALL_FBM(worleyA, 2)
BUILD_ALL_FBM(worleyA, 3)
#endif
#if NOISETYPE == NOISE_WORLEYFB
BUILD_ALL_FBM(worleyB, 2)
BUILD_ALL_FBM(worleyB, 3)
#endif
#if NOISETYPE == NOISE_WHITE
BUILD_ALL_FBM(cellnoise, 2)
BUILD_ALL_FBM(cellnoise, 3)
#endif
#if NOISETYPE == NOISE_TORUS
BUILD_ALL_FBM(gxnoise, 4)
#endif
#if NOISETYPE == NOISE_ALLIGATOR
BUILD_ALL_FBM(alligator, 2)
BUILD_ALL_FBM(alligator, 3)
#endif

#define CALL_CHANNELS(FUNC, POS, PERIOD, CHANNELS) \
    float octaves = AT_inoctaves_bound ? AT_octaves * AT_inoctaves : AT_octaves; \
    float roughness = AT_inroughness_bound ? AT_roughness * AT_inroughness : AT_roughness; \
    \
    if (CHANNELS == 1) \
        result = FUNC##_1(POS, PERIOD, octaves, roughness, AT_lacunarity, AT_contrastexponent, AT_jitter); \
    else if (CHANNELS == 2) \
        result.xy = FUNC##_2(POS, PERIOD, octaves, roughness, AT_lacunarity, AT_contrastexponent, AT_jitter); \
    else if (CHANNELS == 3) \
        result.xyz = FUNC##_3(POS, PERIOD, octaves, roughness, AT_lacunarity, AT_contrastexponent, AT_jitter); \
    else if (CHANNELS == 4) \
        result.xyzw = FUNC##_4(POS, PERIOD, octaves, roughness, AT_lacunarity, AT_contrastexponent, AT_jitter); \


kernel void generickernel( 
    int2 _bound_tilesize,
    float  _bound_amp,
    float  _bound_center,
    float2 _bound_featuresize,
    float2 _bound_off,
    float  _bound_roughness,
    float  _bound_octaves,
    float  _bound_lacunarity,
    float2 _bound_tiledsize,
    int    _bound_istiled,
    int    _bound_noisetype,
    int    _bound_usetime,
    float  _bound_timeoffset,
    float  _bound_timescale,
    float  _bound_parmtime,
    int    _bound_is3d,
    float  _bound_pulselength,
    int    _bound_doloop,
    float  _bound_looptime,
    int    _bound_percomp,
    float  _bound_contrastexponent,
    int    _bound_post_dofold,
    int    _bound_post_docomplement,
    int    _bound_post_dobias,
    float  _bound_post_bias,
    int    _bound_post_dogain,
    float  _bound_post_gain,
    int    _bound_post_dogamma,
    float  _bound_post_gamma,
    int    _bound_post_docontrast,
    float  _bound_post_contrast,
    int    _bound_post_doclampmin,
    float  _bound_post_minimum,
    int    _bound_post_doclampmax,
    float  _bound_post_maximum,
    float  _bound_jitter,
    global void * restrict _bound_noise_stat_void,
    global void * restrict _bound_noise,
#ifdef HAS_src
    global void * restrict _bound_src_stat_void,
    global void * restrict _bound_src,
#else
    float4 _bound_src,
#endif
#ifdef HAS_pos
    global void * restrict _bound_pos_stat_void,
    global void * restrict _bound_pos,
#else
    float2 _bound_pos,
#endif
#ifdef HAS_intime
    global void * restrict _bound_intime_stat_void,
    global void * restrict _bound_intime,
#else
    float _bound_intime,
#endif
#ifdef HAS_inoctaves
    global void * restrict _bound_inoctaves_stat_void,
    global void * restrict _bound_inoctaves,
#else
    float _bound_inoctaves,
#endif
#ifdef HAS_inroughness
    global void * restrict _bound_inroughness_stat_void,
    global void * restrict _bound_inroughness
#else
    float _bound_inroughness
#endif
)
{
    IMX_Layer _bound_noise_layer = {_bound_noise, _bound_noise_stat_void};
#ifdef HAS_src
    IMX_Layer _bound_src_layer = {_bound_src, _bound_src_stat_void};
#endif
#ifdef HAS_pos
    IMX_Layer _bound_pos_layer = {_bound_pos, _bound_pos_stat_void};
#endif
#ifdef HAS_intime
    IMX_Layer _bound_intime_layer = {_bound_intime, _bound_intime_stat_void};
#endif
#ifdef HAS_inoctaves
    IMX_Layer _bound_inoctaves_layer = {_bound_inoctaves, _bound_inoctaves_stat_void};
#endif
#ifdef HAS_inroughness
    IMX_Layer _bound_inroughness_layer = {_bound_inroughness, _bound_inroughness_stat_void};
#endif
    int _bound_gidx = get_global_id(0) * _bound_tilesize.x;
    int _bound_gidy = get_global_id(1) * _bound_tilesize.y;
    if (_bound_gidx >= _RUNOVER_LAYER.stat->resolution.x)
        return;
    if (_bound_gidy >= _RUNOVER_LAYER.stat->resolution.y)
        return;
    int _bound_idx = _linearIndex(_RUNOVER_LAYER.stat, (int2)(_bound_gidx, _bound_gidy));
    float2 _bound_P_image = bufferToImage(_RUNOVER_LAYER.stat, (float2)(_bound_gidx, _bound_gidy));
    float2 _bound_P_texture = bufferToTexture(_RUNOVER_LAYER.stat, (float2)(_bound_gidx, _bound_gidy));
    float2 _bound_P_pixel = bufferToPixel(_RUNOVER_LAYER.stat, (float2)(_bound_gidx, _bound_gidy));

#line 3600

#if AT_pos_bound
    float2 p = AT_pos.xy;
#else
    float2 p = AT_P;
#endif

#if AT_intime_bound
    float lcltime = AT_intime;
#else
    float lcltime = AT_parmtime;
#endif
    lcltime *= AT_timescale;
    lcltime += AT_timeoffset;

    float4 result = 0;
    
    int channels = AT_noise_stat->channels;
    if (AT_percomp == 0)
        channels = 1;

    #if NOISETYPE == NOISE_TORUS
    {
        // Torus noise works in 4d.
        // Tilesize affects conversion to 2*M_PI.
        p -= AT_off;
        p /= AT_tiledsize;
        float4 p4;
        
        // sincos might also be faster, but OTOH
        // sinpi avoids having to round large numbers...
        p4.x = sinpi(p.x * 2);
        p4.y = sinpi(p.x * 2 + 0.5);
        p4.z = sinpi(p.y * 2);
        p4.w = sinpi(p.y * 2  + 0.5);
        
        // One loop of our source goes 2*M_PI in noise space
        // This doesn't work for very large features as you'll
        // orbit a single lattice point
        p4 /= (float)M_PI * 2.0f;
        // Scale to feature size in 4d.
        p4.xy /= AT_featuresize.x;
        p4.zw /= AT_featuresize.y;


        // No 5d noise so have to just offset.
        if (AT_is3d)
        {
            // Normalize speed:
            lcltime /= AT_looptime;
            if (AT_doloop && AT_looptime != 0)
            {
                // We move the torus through noise space along a circle,
                // right angles to the canonical circle.
                // Scale by 2Pi
                float rad = AT_looptime / AT_pulselength;
                rad /= (float) M_PI * 2.0f;
                p4.x += sinpi(2*lcltime+0.5) * rad;
                p4.y -= sinpi(2*lcltime) * rad;
                p4.z += sinpi(2*lcltime+0.5) * rad;
                p4.w -= sinpi(2*lcltime) * rad;
            }
            else
            {
                p4 += (float4)(lcltime,-lcltime,lcltime,-lcltime);
            }
        }
        

        // No need to do any clamping...
        int4 period = 0;
        CALL_CHANNELS(FBMgxnoise_4, p4, period, channels);
    }
    #else
    {
        // 2d noises
        int2    period = 0;
        if (AT_istiled)
        {
            // Clamp to valid size and set period
            float2 fperiod = round(AT_tiledsize / AT_featuresize);
            period.x = (int)fperiod.x;
            period.y = (int)fperiod.y;
            
            if (AT_tiledsize.x == 0)
                period.x = 0;
            else
                AT_featuresize.x = AT_tiledsize.x / fperiod.x;
            if (AT_tiledsize.y == 0)
                period.y = 0;
            else
                AT_featuresize.y = AT_tiledsize.y / fperiod.y;
            AT_lacunarity = round(AT_lacunarity);
        }
        p -= AT_off;
        p /= AT_featuresize;
        
        if (AT_is3d)
        {
            int3 periodtime = (int3)(period.x, period.y, 0);
            float3 p3 = (float3)(p.x, p.y, lcltime);
            
            if (AT_doloop)
            {
                float floopperiod = round(AT_looptime / AT_pulselength);
                
                periodtime.z = (int) floopperiod;
                if (AT_looptime == 0)
                    periodtime.z = 0;
                else
                    AT_pulselength = AT_looptime / floopperiod;
            }
            lcltime /= AT_pulselength;
            p3.z = lcltime;
            
            #if NOISETYPE == NOISE_PERLIN
            {
                CALL_CHANNELS(FBMperlin_3, p3, periodtime, channels);
            }
            #elif NOISETYPE == NOISE_WORLEYFA
            {
                CALL_CHANNELS(FBMworleyA_3, p3, periodtime, channels);
            }
            #elif NOISETYPE == NOISE_WORLEYFB
            {
                CALL_CHANNELS(FBMworleyB_3, p3, periodtime, channels);
            }
            #elif NOISETYPE == NOISE_WHITE
            {
                AT_lacunarity = round(AT_lacunarity);
                CALL_CHANNELS(FBMcellnoise_3, p3, periodtime, channels);
            }
            #elif NOISETYPE == NOISE_ALLIGATOR
            {
                CALL_CHANNELS(FBMalligator_3, p3, periodtime, channels);
            }
            #endif
        }
        else
        {
            #if NOISETYPE == NOISE_PERLIN
            {

                CALL_CHANNELS(FBMperlin_2, p, period, channels);
            }
            #elif NOISETYPE == NOISE_WORLEYFA
            {
                CALL_CHANNELS(FBMworleyA_2, p, period, channels);
            }
            #elif NOISETYPE == NOISE_WORLEYFB
            {
                CALL_CHANNELS(FBMworleyB_2, p, period, channels);
            }
            #elif NOISETYPE == NOISE_WHITE
            {
                AT_lacunarity = round(AT_lacunarity);
                CALL_CHANNELS(FBMcellnoise_2, p, period, channels);
            }
            #elif NOISETYPE == NOISE_ALLIGATOR
            {
                CALL_CHANNELS(FBMalligator_2, p, period, channels);
            }
            #endif
        }
        AT_noise_set(result);
    }
    #endif 
    result *= AT_amp;
    result += AT_center;
    
    result = post_processing_4(result, AT_post_dofold, AT_post_docomplement,
                AT_post_dobias, AT_post_bias, AT_post_dogain, AT_post_gain,
                AT_post_dogamma, AT_post_gamma, AT_post_docontrast, AT_post_contrast,
                AT_post_doclampmin, AT_post_minimum, AT_post_doclampmax, AT_post_maximum);
                
    AT_noise_set(result);
}