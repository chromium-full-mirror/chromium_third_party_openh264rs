// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

// Standalone C11 interop test for the OpenH264 raw C ABI (`c-api`).
// Compiled by a real C compiler against `codec_api.h` and linked against `libopenh264rs`.

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codec_api.h"
#include "codec_app_def.h"
#include "codec_def.h"
#include "codec_ver.h"

#define ASSERT_TRUE(cond)                                                     \
  do {                                                                        \
    if (!(cond)) {                                                            \
      fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, \
              #cond);                                                         \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define ASSERT_EQ(expected, actual)                                           \
  do {                                                                        \
    long long _exp = (long long)(expected);                                   \
    long long _act = (long long)(actual);                                     \
    if (_exp != _act) {                                                       \
      fprintf(stderr,                                                         \
              "ASSERT_EQ failed at %s:%d: expected %s (%lld), got %s (%lld)\n", \
              __FILE__, __LINE__, #expected, _exp, #actual, _act);            \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define ASSERT_GT(val, bound)                                                 \
  do {                                                                        \
    long long _v = (long long)(val);                                          \
    long long _b = (long long)(bound);                                        \
    if (!(_v > _b)) {                                                         \
      fprintf(stderr, "ASSERT_GT failed at %s:%d: %s (%lld) <= %s (%lld)\n",  \
              __FILE__, __LINE__, #val, _v, #bound, _b);                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

enum {
  kSdWidth = 640,
  kSdHeight = 480,
  kNumLoopbackFrames = 15,
};

struct bool_test_struct {
  char c;
  bool b;
};

static void fill_moving_i420_frame(uint8_t *y_plane, uint8_t *u_plane,
                                   uint8_t *v_plane, int width, int height,
                                   int frame_idx) {
  const int box_x = (32 + frame_idx * 12) % (width - 80);
  const int box_y = (24 + frame_idx * 8) % (height - 80);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x >= box_x && x < box_x + 80 && y >= box_y && y < box_y + 80) {
        y_plane[y * width + x] = 220;
      } else {
        int val = 32 + (((x + frame_idx * 6) + (y + frame_idx * 4)) % 160);
        y_plane[y * width + x] = (uint8_t)val;
      }
    }
  }
  const int uv_w = width / 2;
  const int uv_h = height / 2;
  for (int cy = 0; cy < uv_h; ++cy) {
    for (int cx = 0; cx < uv_w; ++cx) {
      u_plane[cy * uv_w + cx] = (uint8_t)(64 + ((cx + frame_idx * 3) % 128));
      v_plane[cy * uv_w + cx] = (uint8_t)(64 + ((cy + frame_idx * 3) % 128));
    }
  }
}

static double plane_psnr(const uint8_t *src, int src_stride,
                         const uint8_t *decoded, int dec_stride, int width,
                         int height) {
  uint64_t sse = 0;
  for (int y = 0; y < height; ++y) {
    const uint8_t *s_row = src + (size_t)y * (size_t)src_stride;
    const uint8_t *d_row = decoded + (size_t)y * (size_t)dec_stride;
    for (int x = 0; x < width; ++x) {
      int diff = (int)s_row[x] - (int)d_row[x];
      sse += (uint64_t)(diff * diff);
    }
  }
  if (sse == 0) {
    return 99.0;
  }
  double mse = (double)sse / (double)(width * height);
  return 10.0 * log10((255.0 * 255.0) / mse);
}

static void test_abi_layout(void) {
  ASSERT_EQ(1, sizeof(bool));
  ASSERT_EQ(1, offsetof(struct bool_test_struct, b));
  ASSERT_EQ(2, sizeof(struct bool_test_struct));

  // Verify vtable slot count and order in the C `ISVCEncoderVtbl` / `ISVCDecoderVtbl`.
  ASSERT_EQ(9 * sizeof(void *), sizeof(ISVCEncoderVtbl));
  ASSERT_EQ(0 * sizeof(void *), offsetof(ISVCEncoderVtbl, Initialize));
  ASSERT_EQ(1 * sizeof(void *), offsetof(ISVCEncoderVtbl, InitializeExt));
  ASSERT_EQ(2 * sizeof(void *), offsetof(ISVCEncoderVtbl, GetDefaultParams));
  ASSERT_EQ(3 * sizeof(void *), offsetof(ISVCEncoderVtbl, Uninitialize));
  ASSERT_EQ(4 * sizeof(void *), offsetof(ISVCEncoderVtbl, EncodeFrame));
  ASSERT_EQ(5 * sizeof(void *), offsetof(ISVCEncoderVtbl, EncodeParameterSets));
  ASSERT_EQ(6 * sizeof(void *), offsetof(ISVCEncoderVtbl, ForceIntraFrame));
  ASSERT_EQ(7 * sizeof(void *), offsetof(ISVCEncoderVtbl, SetOption));
  ASSERT_EQ(8 * sizeof(void *), offsetof(ISVCEncoderVtbl, GetOption));

  ASSERT_EQ(10 * sizeof(void *), sizeof(ISVCDecoderVtbl));
  ASSERT_EQ(0 * sizeof(void *), offsetof(ISVCDecoderVtbl, Initialize));
  ASSERT_EQ(1 * sizeof(void *), offsetof(ISVCDecoderVtbl, Uninitialize));
  ASSERT_EQ(2 * sizeof(void *), offsetof(ISVCDecoderVtbl, DecodeFrame));
  ASSERT_EQ(3 * sizeof(void *), offsetof(ISVCDecoderVtbl, DecodeFrameNoDelay));
  ASSERT_EQ(4 * sizeof(void *), offsetof(ISVCDecoderVtbl, DecodeFrame2));
  ASSERT_EQ(5 * sizeof(void *), offsetof(ISVCDecoderVtbl, FlushFrame));
  ASSERT_EQ(6 * sizeof(void *), offsetof(ISVCDecoderVtbl, DecodeParser));
  ASSERT_EQ(7 * sizeof(void *), offsetof(ISVCDecoderVtbl, DecodeFrameEx));
  ASSERT_EQ(8 * sizeof(void *), offsetof(ISVCDecoderVtbl, SetOption));
  ASSERT_EQ(9 * sizeof(void *), offsetof(ISVCDecoderVtbl, GetOption));
}

static void test_version_and_capability(void) {
  OpenH264Version ver = WelsGetCodecVersion();
  ASSERT_EQ(g_stCodecVersion.uMajor, ver.uMajor);
  ASSERT_EQ(g_stCodecVersion.uMinor, ver.uMinor);
  ASSERT_EQ(g_stCodecVersion.uRevision, ver.uRevision);
  ASSERT_EQ(2u, ver.uMajor);
  ASSERT_EQ(6u, ver.uMinor);
  ASSERT_EQ(0u, ver.uRevision);

  OpenH264Version ver_ex;
  memset(&ver_ex, 0, sizeof(ver_ex));
  WelsGetCodecVersionEx(&ver_ex);
  ASSERT_EQ(2u, ver_ex.uMajor);
  ASSERT_EQ(6u, ver_ex.uMinor);
  ASSERT_EQ(0u, ver_ex.uRevision);

  SDecoderCapability cap;
  memset(&cap, 0, sizeof(cap));
  ASSERT_EQ(cmResultSuccess, WelsGetDecoderCapability(&cap));
  ASSERT_EQ(66, cap.iProfileIdc);
  ASSERT_EQ(32, cap.iLevelIdc);
}

static void test_encoder_lifecycle_and_methods(void) {
  ISVCEncoder *encoder = NULL;
  ASSERT_EQ(cmResultSuccess, WelsCreateSVCEncoder(&encoder));
  ASSERT_TRUE(encoder != NULL);
  ASSERT_TRUE(*encoder != NULL);

  SEncParamExt param;
  memset(&param, 0, sizeof(param));
  ASSERT_EQ(cmResultSuccess, (*encoder)->GetDefaultParams(encoder, &param));

  const int width = kSdWidth;
  const int height = kSdHeight;
  param.iPicWidth = width;
  param.iPicHeight = height;
  param.fMaxFrameRate = 30.0f;
  param.iTargetBitrate = 2000000;
  param.sSpatialLayers[0].iVideoWidth = width;
  param.sSpatialLayers[0].iVideoHeight = height;
  param.sSpatialLayers[0].fFrameRate = 30.0f;
  param.sSpatialLayers[0].iSpatialBitrate = 2000000;
  ASSERT_EQ(cmResultSuccess, (*encoder)->InitializeExt(encoder, &param));

  int trace_level = WELS_LOG_QUIET;
  ASSERT_EQ(cmResultSuccess,
            (*encoder)->SetOption(encoder, ENCODER_OPTION_TRACE_LEVEL,
                                  &trace_level));

  int idr_interval = -1;
  ASSERT_EQ(cmResultSuccess,
            (*encoder)->GetOption(encoder, ENCODER_OPTION_IDR_INTERVAL,
                                  &idr_interval));

  ASSERT_EQ(cmResultSuccess, (*encoder)->ForceIntraFrame(encoder, true));

  SFrameBSInfo ps_info;
  memset(&ps_info, 0, sizeof(ps_info));
  ASSERT_EQ(cmResultSuccess, (*encoder)->EncodeParameterSets(encoder, &ps_info));
  ASSERT_GT(ps_info.iLayerNum, 0);

  size_t y_size = (size_t)(width * height);
  size_t uv_size = (size_t)((width / 2) * (height / 2));
  uint8_t *y_plane = (uint8_t *)malloc(y_size);
  uint8_t *u_plane = (uint8_t *)malloc(uv_size);
  uint8_t *v_plane = (uint8_t *)malloc(uv_size);
  ASSERT_TRUE(y_plane != NULL && u_plane != NULL && v_plane != NULL);
  fill_moving_i420_frame(y_plane, u_plane, v_plane, width, height, 0);

  SSourcePicture src_pic;
  memset(&src_pic, 0, sizeof(src_pic));
  src_pic.iColorFormat = videoFormatI420;
  src_pic.iPicWidth = width;
  src_pic.iPicHeight = height;
  src_pic.iStride[0] = width;
  src_pic.iStride[1] = width / 2;
  src_pic.iStride[2] = width / 2;
  src_pic.pData[0] = y_plane;
  src_pic.pData[1] = u_plane;
  src_pic.pData[2] = v_plane;

  SFrameBSInfo bs_info;
  memset(&bs_info, 0, sizeof(bs_info));
  ASSERT_EQ(cmResultSuccess, (*encoder)->EncodeFrame(encoder, &src_pic, &bs_info));
  ASSERT_EQ(videoFrameTypeIDR, bs_info.eFrameType);
  ASSERT_GT(bs_info.iLayerNum, 0);

  ASSERT_EQ(cmResultSuccess, (*encoder)->Uninitialize(encoder));

  // Re-initialize using SEncParamBase via Initialize()
  SEncParamBase base_param;
  memset(&base_param, 0, sizeof(base_param));
  base_param.iUsageType = CAMERA_VIDEO_REAL_TIME;
  base_param.iPicWidth = width;
  base_param.iPicHeight = height;
  base_param.fMaxFrameRate = 30.0f;
  base_param.iTargetBitrate = 2000000;
  (*encoder)->SetOption(encoder, ENCODER_OPTION_TRACE_LEVEL, &trace_level);
  ASSERT_EQ(cmResultSuccess, (*encoder)->Initialize(encoder, &base_param));
  ASSERT_EQ(cmResultSuccess, (*encoder)->Uninitialize(encoder));

  free(y_plane);
  free(u_plane);
  free(v_plane);
  WelsDestroySVCEncoder(encoder);
}

static uint8_t *flatten_frame_bs(const SFrameBSInfo *bs_info, size_t *out_len) {
  size_t total = 0;
  for (int i = 0; i < bs_info->iLayerNum; ++i) {
    const SLayerBSInfo *layer = &bs_info->sLayerInfo[i];
    for (int j = 0; j < layer->iNalCount; ++j) {
      total += (size_t)layer->pNalLengthInByte[j];
    }
  }
  uint8_t *buf = (uint8_t *)malloc(total > 0 ? total : 1);
  ASSERT_TRUE(buf != NULL);
  size_t pos = 0;
  for (int i = 0; i < bs_info->iLayerNum; ++i) {
    const SLayerBSInfo *layer = &bs_info->sLayerInfo[i];
    size_t layer_len = 0;
    for (int j = 0; j < layer->iNalCount; ++j) {
      layer_len += (size_t)layer->pNalLengthInByte[j];
    }
    memcpy(buf + pos, layer->pBsBuf, layer_len);
    pos += layer_len;
  }
  *out_len = total;
  return buf;
}

static void test_encode_and_decode_loopback(void) {
  const int width = kSdWidth;
  const int height = kSdHeight;

  ISVCEncoder *encoder = NULL;
  ASSERT_EQ(cmResultSuccess, WelsCreateSVCEncoder(&encoder));

  int trace_level = WELS_LOG_QUIET;
  (*encoder)->SetOption(encoder, ENCODER_OPTION_TRACE_LEVEL, &trace_level);

  SEncParamExt param;
  memset(&param, 0, sizeof(param));
  ASSERT_EQ(cmResultSuccess, (*encoder)->GetDefaultParams(encoder, &param));
  param.iPicWidth = width;
  param.iPicHeight = height;
  param.fMaxFrameRate = 30.0f;
  param.iTargetBitrate = 2500000;
  param.iMaxBitrate = 2500000;
  param.bEnableFrameSkip = false;
  param.sSpatialLayers[0].iVideoWidth = width;
  param.sSpatialLayers[0].iVideoHeight = height;
  param.sSpatialLayers[0].fFrameRate = 30.0f;
  param.sSpatialLayers[0].iSpatialBitrate = 2500000;
  param.sSpatialLayers[0].iMaxSpatialBitrate = 2500000;
  ASSERT_EQ(cmResultSuccess, (*encoder)->InitializeExt(encoder, &param));

  size_t y_size = (size_t)(width * height);
  size_t uv_size = (size_t)((width / 2) * (height / 2));
  uint8_t *y_plane = (uint8_t *)malloc(y_size);
  uint8_t *u_plane = (uint8_t *)malloc(uv_size);
  uint8_t *v_plane = (uint8_t *)malloc(uv_size);
  ASSERT_TRUE(y_plane != NULL && u_plane != NULL && v_plane != NULL);

  SSourcePicture src_pic;
  memset(&src_pic, 0, sizeof(src_pic));
  src_pic.iColorFormat = videoFormatI420;
  src_pic.iPicWidth = width;
  src_pic.iPicHeight = height;
  src_pic.iStride[0] = width;
  src_pic.iStride[1] = width / 2;
  src_pic.iStride[2] = width / 2;
  src_pic.pData[0] = y_plane;
  src_pic.pData[1] = u_plane;
  src_pic.pData[2] = v_plane;

  uint8_t *frames[kNumLoopbackFrames];
  size_t frame_lens[kNumLoopbackFrames];
  for (int f = 0; f < kNumLoopbackFrames; ++f) {
    fill_moving_i420_frame(y_plane, u_plane, v_plane, width, height, f);
    src_pic.uiTimeStamp = (long long)(f * 33);
    SFrameBSInfo bs_info;
    memset(&bs_info, 0, sizeof(bs_info));
    ASSERT_EQ(cmResultSuccess,
              (*encoder)->EncodeFrame(encoder, &src_pic, &bs_info));
    if (f == 0) {
      ASSERT_EQ(videoFrameTypeIDR, bs_info.eFrameType);
    } else {
      ASSERT_EQ(videoFrameTypeP, bs_info.eFrameType);
    }
    frames[f] = flatten_frame_bs(&bs_info, &frame_lens[f]);
    ASSERT_GT(frame_lens[f], 0);
  }

  (*encoder)->Uninitialize(encoder);
  WelsDestroySVCEncoder(encoder);

  // Decode all SD frames via ISVCDecoder C vtable and check PSNR against input
  ISVCDecoder *decoder = NULL;
  ASSERT_EQ(cmResultSuccess, WelsCreateDecoder(&decoder));
  ASSERT_TRUE(decoder != NULL);
  ASSERT_TRUE(*decoder != NULL);

  SDecodingParam dec_param;
  memset(&dec_param, 0, sizeof(dec_param));
  dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_SVC;
  ASSERT_EQ(cmResultSuccess, (*decoder)->Initialize(decoder, &dec_param));

  ASSERT_EQ(cmResultSuccess,
            (*decoder)->SetOption(decoder, DECODER_OPTION_TRACE_LEVEL,
                                  &trace_level));

  int eos = 0;
  ASSERT_EQ(cmResultSuccess,
            (*decoder)->GetOption(decoder, DECODER_OPTION_END_OF_STREAM, &eos));
  ASSERT_EQ(0, eos);

  uint8_t *prev_decoded_y = (uint8_t *)malloc(y_size);
  ASSERT_TRUE(prev_decoded_y != NULL);

  // Decode frames 0..kNumLoopbackFrames-2 via DecodeFrameNoDelay, and the final
  // frame via DecodeFrame2 + EOS DecodeFrame2(NULL, 0) / FlushFrame.
  for (int f = 0; f < kNumLoopbackFrames; ++f) {
    unsigned char *dst[3] = {NULL, NULL, NULL};
    SBufferInfo buf_info;
    memset(&buf_info, 0, sizeof(buf_info));
    DECODING_STATE state;
    if (f + 1 < kNumLoopbackFrames) {
      state = (*decoder)->DecodeFrameNoDelay(decoder, frames[f],
                                             (int)frame_lens[f], dst, &buf_info);
      ASSERT_EQ(dsErrorFree, state);
    } else {
      state = (*decoder)->DecodeFrame2(decoder, frames[f], (int)frame_lens[f],
                                       dst, &buf_info);
      ASSERT_EQ(dsErrorFree, state);
      int end_of_stream = 1;
      (*decoder)->SetOption(decoder, DECODER_OPTION_END_OF_STREAM,
                            &end_of_stream);
      if (buf_info.iBufferStatus != 1) {
        state = (*decoder)->DecodeFrame2(decoder, NULL, 0, dst, &buf_info);
        ASSERT_EQ(dsErrorFree, state);
      }
      if (buf_info.iBufferStatus != 1) {
        state = (*decoder)->FlushFrame(decoder, dst, &buf_info);
        ASSERT_EQ(dsErrorFree, state);
      }
    }
    ASSERT_EQ(1, buf_info.iBufferStatus);
    ASSERT_EQ(width, buf_info.UsrData.sSystemBuffer.iWidth);
    ASSERT_EQ(height, buf_info.UsrData.sSystemBuffer.iHeight);
    ASSERT_TRUE(dst[0] != NULL);
    ASSERT_TRUE(dst[1] != NULL);
    ASSERT_TRUE(dst[2] != NULL);

    fill_moving_i420_frame(y_plane, u_plane, v_plane, width, height, f);
    int y_stride = buf_info.UsrData.sSystemBuffer.iStride[0];
    int uv_stride = buf_info.UsrData.sSystemBuffer.iStride[1];
    double psnr_y = plane_psnr(y_plane, width, dst[0], y_stride, width, height);
    double psnr_u =
        plane_psnr(u_plane, width / 2, dst[1], uv_stride, width / 2, height / 2);
    double psnr_v =
        plane_psnr(v_plane, width / 2, dst[2], uv_stride, width / 2, height / 2);
    ASSERT_TRUE(psnr_y >= 30.0);
    ASSERT_TRUE(psnr_u >= 30.0);
    ASSERT_TRUE(psnr_v >= 30.0);

    // Ensure consecutive decoded frames are distinct (motion is preserved).
    if (f > 0) {
      bool diff_found = false;
      for (int row = 0; row < height; ++row) {
        if (memcmp(prev_decoded_y + (size_t)row * (size_t)width,
                   dst[0] + (size_t)row * (size_t)y_stride,
                   (size_t)width) != 0) {
          diff_found = true;
          break;
        }
      }
      ASSERT_TRUE(diff_found);
    }
    for (int row = 0; row < height; ++row) {
      memcpy(prev_decoded_y + (size_t)row * (size_t)width,
             dst[0] + (size_t)row * (size_t)y_stride, (size_t)width);
    }
  }

  ASSERT_EQ(cmResultSuccess, (*decoder)->Uninitialize(decoder));
  WelsDestroyDecoder(decoder);

  free(prev_decoded_y);
  free(y_plane);
  free(u_plane);
  free(v_plane);
  for (int f = 0; f < kNumLoopbackFrames; ++f) {
    free(frames[f]);
  }
}

static void test_decode_parser_mode(void) {
  const int width = kSdWidth;
  const int height = kSdHeight;

  ISVCEncoder *encoder = NULL;
  ASSERT_EQ(cmResultSuccess, WelsCreateSVCEncoder(&encoder));

  SEncParamExt param;
  memset(&param, 0, sizeof(param));
  ASSERT_EQ(cmResultSuccess, (*encoder)->GetDefaultParams(encoder, &param));
  param.iPicWidth = width;
  param.iPicHeight = height;
  param.fMaxFrameRate = 30.0f;
  param.iTargetBitrate = 2000000;
  param.sSpatialLayers[0].iVideoWidth = width;
  param.sSpatialLayers[0].iVideoHeight = height;
  param.sSpatialLayers[0].fFrameRate = 30.0f;
  param.sSpatialLayers[0].iSpatialBitrate = 2000000;
  ASSERT_EQ(cmResultSuccess, (*encoder)->InitializeExt(encoder, &param));

  int trace_level = WELS_LOG_QUIET;
  (*encoder)->SetOption(encoder, ENCODER_OPTION_TRACE_LEVEL, &trace_level);

  size_t y_size = (size_t)(width * height);
  size_t uv_size = (size_t)((width / 2) * (height / 2));
  uint8_t *y_plane = (uint8_t *)malloc(y_size);
  uint8_t *u_plane = (uint8_t *)malloc(uv_size);
  uint8_t *v_plane = (uint8_t *)malloc(uv_size);
  ASSERT_TRUE(y_plane != NULL && u_plane != NULL && v_plane != NULL);
  fill_moving_i420_frame(y_plane, u_plane, v_plane, width, height, 0);

  SSourcePicture src_pic;
  memset(&src_pic, 0, sizeof(src_pic));
  src_pic.iColorFormat = videoFormatI420;
  src_pic.iPicWidth = width;
  src_pic.iPicHeight = height;
  src_pic.iStride[0] = width;
  src_pic.iStride[1] = width / 2;
  src_pic.iStride[2] = width / 2;
  src_pic.pData[0] = y_plane;
  src_pic.pData[1] = u_plane;
  src_pic.pData[2] = v_plane;

  SFrameBSInfo bs_info;
  memset(&bs_info, 0, sizeof(bs_info));
  ASSERT_EQ(cmResultSuccess, (*encoder)->EncodeFrame(encoder, &src_pic, &bs_info));

  size_t bitstream_len = 0;
  uint8_t *bitstream = flatten_frame_bs(&bs_info, &bitstream_len);

  (*encoder)->Uninitialize(encoder);
  WelsDestroySVCEncoder(encoder);
  free(y_plane);
  free(u_plane);
  free(v_plane);

  ISVCDecoder *parser = NULL;
  ASSERT_EQ(cmResultSuccess, WelsCreateDecoder(&parser));
  SDecodingParam parse_param;
  memset(&parse_param, 0, sizeof(parse_param));
  parse_param.bParseOnly = true;
  parse_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_SVC;
  ASSERT_EQ(cmResultSuccess, (*parser)->Initialize(parser, &parse_param));
  (*parser)->SetOption(parser, DECODER_OPTION_TRACE_LEVEL, &trace_level);

  SParserBsInfo parser_info;
  memset(&parser_info, 0, sizeof(parser_info));
  DECODING_STATE state = (*parser)->DecodeParser(
      parser, bitstream, (int)bitstream_len, &parser_info);
  ASSERT_EQ(dsErrorFree, state);

  memset(&parser_info, 0, sizeof(parser_info));
  state = (*parser)->DecodeParser(parser, NULL, 0, &parser_info);
  ASSERT_EQ(dsErrorFree, state);
  ASSERT_GT(parser_info.iNalNum, 0);

  ASSERT_EQ(cmResultSuccess, (*parser)->Uninitialize(parser));
  WelsDestroyDecoder(parser);
  free(bitstream);
}

int main(void) {
  test_abi_layout();
  test_version_and_capability();
  test_encoder_lifecycle_and_methods();
  test_encode_and_decode_loopback();
  test_decode_parser_mode();
  printf("c_api_test: all 5 tests passed\n");
  return 0;
}
