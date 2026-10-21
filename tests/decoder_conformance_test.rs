// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE_CHROMIUM file.

#[path = "common/mod.rs"]
mod common;
use common::Sha1Hasher;
use openh264rs::api::codec_api::*;
use openh264rs::split_annexb_units;

fn update_hash_from_frame(hasher: &mut Sha1Hasher, dec: &Decoder, buf_info: &SBufferInfo) {
    for row in buf_info
        .y_rows(dec)
        .chain(buf_info.u_rows(dec))
        .chain(buf_info.v_rows(dec))
    {
        hasher.update(row);
    }
}

#[test]
pub(crate) fn test_sha1_hasher_sanity() {
    let mut hasher = Sha1Hasher::new();
    hasher.update(b"abc");
    assert_eq!(hasher.digest(), "a9993e364706816aba3e25717850c26c9cd0d89d");
}

#[test]
pub(crate) fn test_decoder_capability_query() {
    let dec_cap = Decoder::capability();
    assert_eq!(dec_cap.iProfileIdc, 66);
    assert_eq!(dec_cap.iProfileIop, 0xE0);
    assert_eq!(dec_cap.iLevelIdc, 32);
    assert_eq!(dec_cap.iMaxMbps, 216000);
    assert_eq!(dec_cap.iMaxFs, 5120);
    assert_eq!(dec_cap.iMaxCpb, 20000);
    assert_eq!(dec_cap.iMaxDpb, 20480);
    assert_eq!(dec_cap.iMaxBr, 20000);
    assert!(!dec_cap.bRedPicCap);
}

/// Decodes one asset and compares the SHA-1 of its output planes against the
/// C++ decoder's.
///
/// `hash_concealed` selects which frames count: `false` hashes only frames returned
/// `dsErrorFree`, `true` hashes every frame written out (`iBufferStatus == 1`,
/// whatever the decoding state). A stream that deliberately conceals needs `true`,
/// or its concealed frames drop out of the hash.
fn test_single_bitstream_asset_ex(file_name: &str, expected_hash: &str, hash_concealed: bool) {
    let data = common::read_stream(file_name);
    assert!(!data.is_empty(), "Bitstream asset {} is empty", file_name);

    let mut decoder = Decoder::new();

    let mut dec_param = SDecodingParam::default();
    dec_param.uiTargetDqLayer = u8::MAX;
    dec_param.eEcActiveIdc = ERROR_CON_IDC::ERROR_CON_SLICE_COPY;
    dec_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

    let init_ret = decoder.initialize(&dec_param);
    assert_eq!(
        init_ret, CM_RESULT_SUCCESS as std::ffi::c_long,
        "Failed to initialize decoder for {}",
        file_name
    );

    let mut hasher = Sha1Hasher::new();
    let units = split_annexb_units(&data);
    let mut decoded_frames = 0;

    for unit in units {
        let mut buf_info = SBufferInfo::default();
        let dec_ret = decoder.decode(Some(unit), &mut buf_info);
        if (hash_concealed || dec_ret == DECODING_STATE::dsErrorFree) && buf_info.iBufferStatus == 1
        {
            update_hash_from_frame(&mut hasher, &decoder, &buf_info);
            decoded_frames += 1;
        }

        // A null `DecodeFrame2` after every NAL (`h264dec.cpp:418-430`)
        // constructs the access unit the NAL completed instead of leaving it
        // pending. That is observable: a same-id PPS arriving next is parked in
        // the spare slot for `WriteBackActiveParameters` rather than overwriting
        // `sPpsBuffer[id]` in place under the pending picture
        // (`au_parser.cpp:1458` against `:1465`).
        let mut buf_info = SBufferInfo::default();
        let dec_ret = decoder.decode(None, &mut buf_info);
        if (hash_concealed || dec_ret == DECODING_STATE::dsErrorFree) && buf_info.iBufferStatus == 1
        {
            update_hash_from_frame(&mut hasher, &decoder, &buf_info);
            decoded_frames += 1;
        }
    }

    // Flush remaining frames in decoder buffer
    decoder.set_end_of_stream(true);

    let mut buf_info = SBufferInfo::default();
    let dec_ret = decoder.decode(None, &mut buf_info);
    if (hash_concealed || dec_ret == DECODING_STATE::dsErrorFree) && buf_info.iBufferStatus == 1 {
        update_hash_from_frame(&mut hasher, &decoder, &buf_info);
        decoded_frames += 1;
    }

    let remaining_frames = decoder.frames_remaining();

    for _f in 0..remaining_frames {
        let mut buf_info = SBufferInfo::default();
        let flush_ret = decoder.flush(&mut buf_info);
        if (hash_concealed || flush_ret == DECODING_STATE::dsErrorFree)
            && buf_info.iBufferStatus == 1
        {
            update_hash_from_frame(&mut hasher, &decoder, &buf_info);
            decoded_frames += 1;
        }
    }

    assert!(decoded_frames > 0, "No frames decoded for {}", file_name);
    let calculated_hash = hasher.digest();
    // The frame count rides in the message: a count that also moved is a
    // different defect from one that held.
    assert_eq!(
        calculated_hash, expected_hash,
        "SHA-1 hash mismatch for bitstream asset {} ({decoded_frames} frames decoded)",
        file_name
    );

    decoder.uninitialize();
}

#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ba_mw_d() {
    test_single_bitstream_asset_ex(
        "res/BA_MW_D.264",
        "afd7a9765961ca241bb4bdf344b31397bec7465a",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_adobe_pdf_sample_a_1024x768_50frms() {
    test_single_bitstream_asset_ex(
        "res/Adobe_PDF_sample_a_1024x768_50Frms.264",
        "9aa9a4d9598eb3e1093311826844f37c43e4c521",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ba1_ft_c() {
    test_single_bitstream_asset_ex(
        "res/BA1_FT_C.264",
        "418d152fb85709b6f172799dcb239038df437cfa",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ba1_sony_d() {
    // I slices, one slice per picture. Deblocking enabled.
    test_single_bitstream_asset_ex(
        "res/BA1_Sony_D.jsv",
        "d94b5ceed5686a03ea682b53d415dee999d27eb6",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ba2_sony_f() {
    // I/P slices, one slice per picture. Deblocking enabled. POC type 0.
    test_single_bitstream_asset_ex(
        "res/BA2_Sony_F.jsv",
        "134e864d7b6f24d0cbc0331c512788c27d8aba4e",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_bamq1_jvc_c() {
    test_single_bitstream_asset_ex(
        "res/BAMQ1_JVC_C.264",
        "613cf662c23e5d9e1d7da7fe880a3c427411d171",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_bamq2_jvc_c() {
    test_single_bitstream_asset_ex(
        "res/BAMQ2_JVC_C.264",
        "11bcf3713f520e606a8326d37e00e5fd6c9fd4a0",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_banm_mw_d() {
    test_single_bitstream_asset_ex(
        "res/BANM_MW_D.264",
        "92d924a857a1a7d7d9b224eaa3887830f15dee7f",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_basqp1_sony_c() {
    test_single_bitstream_asset_ex(
        "res/BASQP1_Sony_C.jsv",
        "3986c8c9d2876d2f0748b925101b152c6ec8b811",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_caba2_sva_b() {
    // I/P slices, one slice per picture. Deblocking enabled. CABAC. POC type 0.
    // num_ref_frames=5.
    test_single_bitstream_asset_ex(
        "res/CABA2_SVA_B.264",
        "3ce8754cf9ab0372dee4de37f3a851d80f5cd76f",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_caba3_sva_b() {
    // IPB slices with CABAC. Temporal direct prediction. num_ref_frames=5.
    test_single_bitstream_asset_ex(
        "res/CABA3_SVA_B.264",
        "1b811ad90ac4ce9beb034cb8c993ad4e2935f46a",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cabast3_sony_e() {
    // 4 slices per picture at first_mb_in_slice 0 / 99 / 198 / 297, with the slice
    // types changing inside the picture — I/P/I/P in the reference pictures, I/P/B/I
    // in the non-reference ones — and deblocking across the boundaries between them.
    // 25 pictures. CABAC. IPB slices, POC type 0, 1 ref frame, temporal direct.
    test_single_bitstream_asset_ex(
        "res/CABAST3_Sony_E.jsv",
        "cb772948a52c128971dd2eaa28d0a66fc7e7a733",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_canl1_sony_e() {
    // I slices, one slice per picture. Deblocking disabled. CABAC. POC type 0.
    test_single_bitstream_asset_ex(
        "res/CANL1_Sony_E.jsv",
        "ba24ac54c8b39a648e7c676c3d9d0160d635830b",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_canl1_toshiba_g() {
    // I slices, one slice per picture. Deblocking disabled. CABAC. POC type 2.
    test_single_bitstream_asset_ex(
        "res/CANL1_TOSHIBA_G.264",
        "a6ce8c15a4e0c4a553c3f8212d3976ff4b90ac18",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_canl2_sony_e() {
    // I/P slices, one slice per picture. Deblocking disabled. CABAC. POC type 0.
    test_single_bitstream_asset_ex(
        "res/CANL2_Sony_E.jsv",
        "cea24d25144d72216301f48c307a21f660c97136",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ci1_ft_b() {
    test_single_bitstream_asset_ex(
        "res/CI1_FT_B.264",
        "cbfec15e17a504678b19a1191992131c92a1ac26",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ci_mw_d() {
    test_single_bitstream_asset_ex(
        "res/CI_MW_D.264",
        "289f29a103c8d95adf2909c646466904be8b06d7",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cvbs3_sony_c() {
    // IPB slices with CAVLC. Temporal direct prediction. direct_8x8_inference=on. num_ref_frames=4.
    test_single_bitstream_asset_ex(
        "res/CVBS3_Sony_C.jsv",
        "094e116aab7b1dd8b16279482ad176b67004a285",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cvfc1_sony_c() {
    test_single_bitstream_asset_ex(
        "res/CVFC1_Sony_C.jsv",
        "4641abd7419a5580b97f16e83fd1d566339229d0",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ls_sva_d() {
    test_single_bitstream_asset_ex(
        "res/LS_SVA_D.264",
        "72118f4d1674cf14e58bed7e67cb3aeed3df62b9",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_midr_mw_d() {
    test_single_bitstream_asset_ex(
        "res/MIDR_MW_D.264",
        "9467030f4786f75644bf06a7fc809c36d1959827",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_mps_mw_a() {
    test_single_bitstream_asset_ex(
        "res/MPS_MW_A.264",
        "67f1cfbef0e8025ed60dedccf8d9558d0636be5f",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_mr1_bt_a() {
    test_single_bitstream_asset_ex(
        "res/MR1_BT_A.h264",
        "6e585f8359667a16b03e5f49a06f5ceae8d991e0",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_mr1_mw_a() {
    test_single_bitstream_asset_ex(
        "res/MR1_MW_A.264",
        "d9e2bf34e9314dcc171ddaea2c5015d0421479f2",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_mr2_mw_a() {
    test_single_bitstream_asset_ex(
        "res/MR2_MW_A.264",
        "628b1d4eff04c2d277f7144e23484957dad63cbe",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_mr2_tandberg_e() {
    test_single_bitstream_asset_ex(
        "res/MR2_TANDBERG_E.264",
        "74d618bc7d9d41998edf4c85d51aa06111db6609",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_nl1_sony_d() {
    // I slices, one slice per picture. Deblocking disabled
    // (disable_deblocking_filter_idc=1).
    test_single_bitstream_asset_ex(
        "res/NL1_Sony_D.jsv",
        "e401e30669938443c2f02522fd4d5aa1382931a0",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_nl2_sony_h() {
    // I/P slices, one slice per picture. Deblocking disabled. POC type 0.
    // Constrained Baseline, yuv420p progressive, 176x144.
    test_single_bitstream_asset_ex(
        "res/NL2_Sony_H.jsv",
        "c68a79d2056fe78279775414b532cd58dc2d9318",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_nlmq1_jvc_c() {
    test_single_bitstream_asset_ex(
        "res/NLMQ1_JVC_C.264",
        "f3265c6ddf8db1b2bf604d8a2954f75532e28cda",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_nlmq2_jvc_c() {
    test_single_bitstream_asset_ex(
        "res/NLMQ2_JVC_C.264",
        "350ae86ef9ba09390d63a09b7f9ff54184109ca8",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_nrf_mw_e() {
    test_single_bitstream_asset_ex(
        "res/NRF_MW_E.264",
        "20732198c04cd2591350a361e4510892f6eed3f0",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_qcif_2p_i_allipcm() {
    test_single_bitstream_asset_ex(
        "res/QCIF_2P_I_allIPCM.264",
        "8724c0866ebdba7ebb7209a0c0c3ae3ae38a0240",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sva_ba1_b() {
    test_single_bitstream_asset_ex(
        "res/SVA_BA1_B.264",
        "c4543b24823b16c424c673616c36c7f537089b2d",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sva_ba2_d() {
    // I/P slices, one slice per picture. Deblocking enabled. POC type 2.
    test_single_bitstream_asset_ex(
        "res/SVA_BA2_D.264",
        "98ff2d67860462d8d8bcc9352097c06cc401d97e",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sva_base_b() {
    // 3 slices per picture. CAVLC. I/P slices, POC type 2, 5 ref frames.
    // disable_deblocking_filter_idc=0, so deblocking filters across slice boundaries.
    test_single_bitstream_asset_ex(
        "res/SVA_Base_B.264",
        "91f514d81cd33de9f6fbf5dbefdb189cc2e7ecf4",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sva_cl1_e() {
    test_single_bitstream_asset_ex(
        "res/SVA_CL1_E.264",
        "4fe09ab6cdc965ea10a20f1d6dd38aca954412bb",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sva_fm1_e() {
    test_single_bitstream_asset_ex(
        "res/SVA_FM1_E.264",
        "fad08c4ff7cf2307b6579853d0f4652fc26645d3",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sva_nl1_b() {
    // I slices, one slice per picture. Deblocking disabled
    // (disable_deblocking_filter_idc=1).
    test_single_bitstream_asset_ex(
        "res/SVA_NL1_B.264",
        "6d63f72a0c0d833b1db0ba438afff3b4180fb3e6",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sva_nl2_e() {
    test_single_bitstream_asset_ex(
        "res/SVA_NL2_E.264",
        "70453ef8097c94dd190d6d2d1d5cb83c67e66238",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sarvui() {
    test_single_bitstream_asset_ex(
        "res/SarVui.264",
        "98ff2d67860462d8d8bcc9352097c06cc401d97e",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_static() {
    test_single_bitstream_asset_ex(
        "res/Static.264",
        "91dd4a7a796805b2cd015cae8fd630d96c663f42",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_zhling_1280x720() {
    test_single_bitstream_asset_ex(
        "res/Zhling_1280x720.264",
        "ad99f5eaa2d73ae3840e7da67313de8cfc866ce6",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_sps_subsetsps_bothvui() {
    test_single_bitstream_asset_ex(
        "res/sps_subsetsps_bothVUI.264",
        "d3a47032eb5dcc1963343a68e9bea12435bf1e4c",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_test_cif_p_cabac_slice() {
    test_single_bitstream_asset_ex(
        "res/test_cif_P_CABAC_slice.264",
        "521bbd0ba2422369b724c7054545cf107a56f959",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_test_qcif_cabac() {
    test_single_bitstream_asset_ex(
        "res/test_qcif_cabac.264",
        "587d1d05943f3cd416bf69469975fdee05361e69",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_test_scalinglist_jm() {
    test_single_bitstream_asset_ex(
        "res/test_scalinglist_jm.264",
        "f690a3af2896a53360215fb5d35016bfd41499b3",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_test_vd_1d() {
    test_single_bitstream_asset_ex(
        "res/test_vd_1d.264",
        "5827d2338b79ff82cd091c707823e466197281d3",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_test_vd_rc() {
    test_single_bitstream_asset_ex(
        "res/test_vd_rc.264",
        "eea02e97bfec89d0418593a8abaaf55d02eaa1ca",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cisco_men_whisper_640x320_cabac_bframe_9() {
    test_single_bitstream_asset_ex(
        "res/Cisco_Men_whisper_640x320_CABAC_Bframe_9.264",
        "2b349c1bc806b6e0412008747b2463d77b576476",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cisco_men_whisper_640x320_cavlc_bframe_9() {
    test_single_bitstream_asset_ex(
        "res/Cisco_Men_whisper_640x320_CAVLC_Bframe_9.264",
        "e5b76ff7e2f44e9b33906f8a4039d0d2bdb1580b",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cisco_adobe_pdf_sample_a_1024x768_cavlc_bframe_9() {
    test_single_bitstream_asset_ex(
        "res/Cisco_Adobe_PDF_sample_a_1024x768_CAVLC_Bframe_9.264",
        "9d758d9e6f4dead0d7b361f3ddf2ee009d0ea190",
        false,
    );
}

// ---------------------------------------------------------------------------
// Narrow frames
//
// `ExpandReferencingPicture` takes a different arm for `iWidth >> 1 < 16`, i.e.
// a frame narrower than 32 luma pixels. The streams below have non-zero MVs
// pointing outside the frame, the only way an expanded border reaches the output.
//
//  * 16x16 — the minimum legal frame width; `iWidthUV` 8, the divergent arm.
//  * 24x18 — coded 32x32 and cropped, so `iWidthUV` is exactly 16: the other
//    side of the same branch, plus frame cropping.
//  * 16x16 with a lost IDR — reaches `WelsInitRefList`'s error-concealment
//    prefetch.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_narrow_16x16() {
    test_single_bitstream_asset_ex(
        "res/narrow_16x16.264",
        "6299ce8a7dc8a86d367dca65ca123eb499fc5ca8",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_narrow_24x18() {
    test_single_bitstream_asset_ex(
        "res/narrow_24x18.264",
        "f6197477215d8847b570982d3c2747da2911f047",
        false,
    );
}
// Two 16x16 encodes concatenated, the second one's IDR NAL removed: a CAVLC
// (profile_idc 66) sequence of 24 frames, then a CABAC (profile_idc 100) SPS,
// which differs from the stored one and so begins a new sequence, and then that
// sequence's P slices with no IDR to open them. The new sequence clears the
// reference lists, the first P slice therefore finds them empty, and
// `WelsCheckAndRecoverForFutureDecoding` prefetches a *recycled* picture — one
// still holding the first sequence's samples outside the area it memsets to 128
// — and expands its border.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_narrow_16x16_idr_lost() {
    test_single_bitstream_asset_ex(
        "res/narrow_16x16_idr_lost.264",
        "754db24b395cc7aff338e036a416a9b5bb409c81",
        true,
    );
}

// ---------------------------------------------------------------------------
// The macroblock-grid probe
//
// 48x32 is 3x2 macroblocks — the smallest grid holding a macroblock with all four
// neighbours as well as one missing only its left and one missing only its
// top-right. CABAC, High profile with the 8x8 transform, I/P/B slices, and a
// panned source window so the MVs are non-zero.
//
// Built with ffmpeg/libx264 because OpenH264's encoder has no
// `transform_8x8_mode_flag` to write.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_grid_48x32() {
    test_single_bitstream_asset_ex(
        "res/grid_48x32.264",
        "a56242a64a22edee058cd9cd14179f54bcd79b97",
        false,
    );
}

// ---------------------------------------------------------------------------
// Error concealment
//
// `BA_MW_D_P_LOST` is the conformance `BA_MW_D` with a P slice dropped, so each
// affected picture is decoded with a hole, under `ERROR_CON_SLICE_COPY`.
//
// `_IDR_LOST`: with the IDR gone there is no complete non-ECed IDR to clear
// `bFreezeOutput`.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ba_mw_d_idr_lost() {
    test_single_bitstream_asset_ex(
        "res/BA_MW_D_IDR_LOST.264",
        "f31e54a50296406f7f55d95c959e800d5dfbbe44",
        true,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_ba_mw_d_p_lost() {
    test_single_bitstream_asset_ex(
        "res/BA_MW_D_P_LOST.264",
        "66ec2f1d66356de3dfadffa298bbe838347a9eb1",
        true,
    );
}

// ---------------------------------------------------------------------------
// Flexible macroblock ordering
//
// 64x64, two slice groups interleaved 1-1 (`slice_group_map_type` 0), one slice per
// group, every macroblock I_PCM with a distinct flat luma value. Slice 0 walks
// macroblocks 0, 2, 4, … and slice 1 walks 1, 3, 5, …; a raster-order `FmoNextMb`
// walks 0, 1, 2, … instead, so every macroblock after each slice's first lands in
// the wrong place. I_PCM makes the frame be the coded values, so a failure is
// legible rather than a diffuse hash change.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_fmo_2groups_64x64() {
    test_single_bitstream_asset_ex(
        "res/fmo_2groups_64x64.264",
        "6420bae4a88f86a0f3f54c94aee59b2c1e7b7319",
        false,
    );
}

// ---------------------------------------------------------------------------
// Custom 8x8 scaling lists
//
// `WelsCalcDeqCoeffScalingList` (`decode_slice.cpp:1486`) is the decoder's only
// reader of `g_kuiMatrixV` (`common_tables.cpp:64`): it builds
// `pDequant_coeff8x8[i][q][y]` as `iScalingList8x8[i][y] * g_kuiMatrixV[q % 6][y /
// 8][y % 8]`. The table is therefore reachable only from a stream that both
// signals a scaling matrix and codes 8x8 transform blocks; `test_scalinglist_jm.264`
// above signals one with `transform_8x8_mode_flag = 0`, so it dequantizes 4x4 only.
//
// Both assets are built by ffmpeg/libx264 (`-x264-params 8x8dct=1:cqm=jvt`), since
// OpenH264's encoder has no `transform_8x8_mode_flag` to write. x264 carries
// `cqm=jvt` in the PPS (`pic_scaling_matrix_present_flag = 1`) and leaves lists 6
// and 7 to the fall-back rule, which is what makes them non-flat.
//
//  * `_intra` — all-intra, CABAC: list 6 through `ParseResidualBlockCabac8x8`
//    (`parse_mb_syn_cabac.cpp:1399`).
//  * `_inter` — I+P, CAVLC: lists 6 and 7 through `WelsResidualBlockCavlc8x8`
//    (`parse_mb_syn_cavlc.cpp:979`).
//  * `_cabac_inter` — I+P+B, CABAC: lists 6 and 7 through
//    `ParseResidualBlockCabac8x8`.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cqm8x8_intra_176x144() {
    test_single_bitstream_asset_ex(
        "res/cqm8x8_intra_176x144.264",
        "42ee8dab05d3eff2e31219b2cf29ee9a8d1e1454",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cqm8x8_inter_176x144() {
    test_single_bitstream_asset_ex(
        "res/cqm8x8_inter_176x144.264",
        "af78d465b33871858868f29eb61b79e7a4c5e5e1",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cqm8x8_cabac_inter_176x144() {
    test_single_bitstream_asset_ex(
        "res/cqm8x8_cabac_inter_176x144.264",
        "15745e18965e4169b8d77feb2a3ac7fdb60a177a",
        false,
    );
}

// ---------------------------------------------------------------------------
// High-profile 8x8 transform and weighted prediction across multiple slices
//
//  * `high_8x8_multislice_176x144` — High profile (`transform_8x8_mode_flag = 1`),
//    CABAC, I/P/B slices, 3 slices per picture (`-x264-params slices=3:8x8dct=1`),
//    filtering 8x8 transform macroblocks across slice boundaries.
//  * `multislice_weighted_176x144` — Main profile, CABAC, I/P/B slices, 3 slices per
//    picture with explicit P-slice (`weighted_pred_flag = 1`) and implicit B-slice
//    (`weighted_bipred_idc = 2`) weighted prediction (`slices=3:weightp=2:weightb=1`),
//    where each slice carries its own `pred_weight_table`.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_high_8x8_multislice_176x144() {
    test_single_bitstream_asset_ex(
        "res/high_8x8_multislice_176x144.264",
        "e1f880c0a650e08a5bc7675dfb488d4ef47ea655",
        false,
    );
}
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_multislice_weighted_176x144() {
    test_single_bitstream_asset_ex(
        "res/multislice_weighted_176x144.264",
        "dea28402c0f14935d2cfd1bc4c28c6ed654ff7c4",
        false,
    );
}

// ---------------------------------------------------------------------------
// A same-id PPS replaced before every picture
//
// `CACQP3_Sony_D.jsv` sends a PPS with `pic_parameter_set_id = 0` ahead of each
// of its 50 pictures, stepping `chroma_qp_index_offset` from -12 to +12. Which
// offset a picture is reconstructed with depends on whether its access unit is
// still pending when the next PPS is parsed: pending, and `pCtx->pPps` is null,
// so `au_parser.cpp:1465` overwrites `sPpsBuffer[0]` in place under it;
// constructed, and `au_parser.cpp:1458` parks the new PPS in the spare slot for
// `WriteBackActiveParameters` (`decoder_core.cpp:2216`) to install afterwards.
//
// The harness's null `DecodeFrame2` after every NAL decides which of the two paths
// each picture takes.
#[test]
#[cfg_attr(not(has_test_data), ignore = "test input files not found")]
pub(crate) fn test_asset_cacqp3_sony_d() {
    test_single_bitstream_asset_ex(
        "res/CACQP3_Sony_D.jsv",
        "2ab41abd239c1c556e91ff027f47002001ee8605",
        false,
    );
}
