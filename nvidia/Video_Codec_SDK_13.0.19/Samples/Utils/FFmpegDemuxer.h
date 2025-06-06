/*
 * This copyright notice applies to this header file only:
 *
 * Copyright (c) 2010-2024 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the software, and to permit persons to whom the
 * software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
/* Explicitly include bsf.h when building against FFmpeg 4.3 (libavcodec 58.45.100) or later for backward compatibility */
#if LIBAVCODEC_VERSION_INT >= 3824484
#include <libavcodec/bsf.h>
#endif
}
#include "NvCodecUtils.h"

//---------------------------------------------------------------------------
//! \file FFmpegDemuxer.h 
//! \brief Provides functionality for stream demuxing
//!
//! This header file is used by Decode/Transcode apps to demux input video clips before decoding frames from it. 
//---------------------------------------------------------------------------

/**
* @brief libavformat wrapper class. Retrieves the elementary encoded stream from the container format.
*/
class FFmpegDemuxer {
private:
    AVFormatContext *fmtc = NULL;
    AVIOContext *avioc = NULL;
    AVPacket* pkt = NULL; /*!< AVPacket stores compressed data typically exported by demuxers and then passed as input to decoders */
    AVPacket* pktFiltered = NULL;
    AVBSFContext *bsfc = NULL;

    int iVideoStream;
    bool bMp4H264, bMp4HEVC, bMp4MPEG4;
    AVCodecID eVideoCodec;
    AVPixelFormat eChromaFormat;
    int nWidth, nHeight, nBitDepth, nBPP, nChromaHeight;
    double timeBase = 0.0;
    int64_t userTimeScale = 0; 

    uint8_t *pDataWithHeader = NULL;

    unsigned int frameCount = 0;

public:
    class DataProvider {
    public:
        virtual ~DataProvider() {}
        virtual int GetData(uint8_t *pBuf, int nBuf) = 0;
    };

private:
    /**
    *   @brief  Private constructor to initialize libavformat resources.
    *   @param  fmtc - Pointer to AVFormatContext allocated inside avformat_open_input()
    */
    FFmpegDemuxer(AVFormatContext *fmtc, int64_t timeScale = 1 /*Hz*/) : fmtc(fmtc) {
        if (!fmtc) {
            LOG(ERROR) << "No AVFormatContext provided.";
            return;
        }

        // Allocate the AVPackets and initialize to default values
        pkt = av_packet_alloc();
        pktFiltered = av_packet_alloc();
        if (!pkt || !pktFiltered) {
            LOG(ERROR) << "AVPacket allocation failed";
            return;
        }

        LOG(INFO) << "Media format: " << fmtc->iformat->long_name << " (" << fmtc->iformat->name << ")";

        ck(avformat_find_stream_info(fmtc, NULL));
        iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (iVideoStream < 0) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "Could not find stream in input file";
            av_packet_free(&pkt);
            av_packet_free(&pktFiltered);
            return;
        }

        //fmtc->streams[iVideoStream]->need_parsing = AVSTREAM_PARSE_NONE;
        eVideoCodec = fmtc->streams[iVideoStream]->codecpar->codec_id;
        nWidth = fmtc->streams[iVideoStream]->codecpar->width;
        nHeight = fmtc->streams[iVideoStream]->codecpar->height;
        eChromaFormat = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->format;
        AVRational rTimeBase = fmtc->streams[iVideoStream]->time_base;
        timeBase = av_q2d(rTimeBase);
        userTimeScale = timeScale;

        // Set bit depth, chroma height, bits per pixel based on eChromaFormat of input
        switch (eChromaFormat)
        {
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_GRAY10LE:   // monochrome is treated as 420 with chroma filled with 0x0
            nBitDepth = 10;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV420P12LE:
            nBitDepth = 12;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P10LE:
            nBitDepth = 10;
            nChromaHeight = nHeight << 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P12LE:
            nBitDepth = 12;
            nChromaHeight = nHeight << 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P:
            nBitDepth = 8;
            nChromaHeight = nHeight << 1;
            nBPP = 1;
            break;
        case AV_PIX_FMT_YUV422P10LE:
            nBitDepth = 10;
            nChromaHeight = nHeight;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV422P12LE:
            nBitDepth = 12;
            nChromaHeight = nHeight;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV422P:
            nBitDepth = 8;
            nChromaHeight = nHeight;
            nBPP = 1;
            break;
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_YUVJ422P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
        case AV_PIX_FMT_YUVJ444P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
        case AV_PIX_FMT_GRAY8:      // monochrome is treated as 420 with chroma filled with 0x0
            nBitDepth = 8;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 1;
            break;
        default:
            LOG(WARNING) << "ChromaFormat not recognized. Assuming 420";
            eChromaFormat = AV_PIX_FMT_YUV420P;
            nBitDepth = 8;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 1;
        }

        bMp4H264 = eVideoCodec == AV_CODEC_ID_H264 && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV") 
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") 
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );
        bMp4HEVC = eVideoCodec == AV_CODEC_ID_HEVC && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );

        bMp4MPEG4 = eVideoCodec == AV_CODEC_ID_MPEG4 && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );

        // Initialize bitstream filter and its required resources
        if (bMp4H264) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf) {
                LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "av_bsf_get_by_name() failed";
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                return;
            }
            ck(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            ck(av_bsf_init(bsfc));
        }
        if (bMp4HEVC) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            if (!bsf) {
                LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " " << "av_bsf_get_by_name() failed";
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                return;
            }
            ck(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            ck(av_bsf_init(bsfc));
        }
    }

    AVFormatContext *CreateFormatContext(DataProvider *pDataProvider) {

        AVFormatContext *ctx = NULL;
        if (!(ctx = avformat_alloc_context())) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }

        uint8_t *avioc_buffer = NULL;
        int avioc_buffer_size = 32 * 1024 * 1024;
        avioc_buffer = (uint8_t *)av_malloc(avioc_buffer_size);
        if (!avioc_buffer) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }
        avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size,
            0, pDataProvider, &ReadPacket, NULL, NULL);
        if (!avioc) {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return NULL;
        }
        ctx->pb = avioc;

        ck(avformat_open_input(&ctx, NULL, NULL, NULL));
        return ctx;
    }

    /**
    *   @brief  Allocate and return AVFormatContext*.
    *   @param  szFilePath - Filepath pointing to input stream.
    *   @return Pointer to AVFormatContext
    */
     AVFormatContext *CreateFormatContext(const char *szFilePath) {
        avformat_network_init();

        AVFormatContext *ctx = NULL;
        ck(avformat_open_input(&ctx, szFilePath, NULL, NULL));
        return ctx;
    }

public:
    FFmpegDemuxer(const char *szFilePath, int64_t timescale = 1 /*Hz*/) : FFmpegDemuxer(CreateFormatContext(szFilePath), timescale) {}
    FFmpegDemuxer(DataProvider *pDataProvider) : FFmpegDemuxer(CreateFormatContext(pDataProvider)) {avioc = fmtc->pb;}
    ~FFmpegDemuxer() {

        if (!fmtc) {
            return;
        }

        if (pkt) {
            av_packet_free(&pkt);
        }
        if (pktFiltered) {
            av_packet_free(&pktFiltered);
        }

        if (bsfc) {
            av_bsf_free(&bsfc);
        }

        avformat_close_input(&fmtc);

        if (avioc) {
            av_freep(&avioc->buffer);
            av_freep(&avioc);
        }

        if (pDataWithHeader) {
            av_free(pDataWithHeader);
        }
    }
    AVFormatContext* GetAVFormatContext() {
        return fmtc;
    }
    AVCodecID GetVideoCodec() {
        return eVideoCodec;
    }
    AVPixelFormat GetChromaFormat() {
        return eChromaFormat;
    }
    int GetWidth() {
        return nWidth;
    }
    int GetHeight() {
        return nHeight;
    }
    int GetBitDepth() {
        return nBitDepth;
    }
    int GetFrameSize() {
        return nWidth * (nHeight + nChromaHeight) * nBPP;
    }
    bool Demux(uint8_t **ppData, int *pnBytes, int64_t *pts = NULL, int64_t* dts = NULL, int *isVideoPacket = NULL, int *streamIndex = NULL) {
        if (!fmtc) {
            return false;
        }

        *pnBytes = 0;

        if (pkt->data) {
            av_packet_unref(pkt);
        }

        if (av_read_frame(fmtc, pkt) < 0) {
            if (isVideoPacket)
                *isVideoPacket = 1;
            if (streamIndex)
                *streamIndex = iVideoStream;
            return false;
        }

        if (pkt->stream_index == iVideoStream)
        {
            if (bMp4H264 || bMp4HEVC) {
                if (pktFiltered->data) {
                    av_packet_unref(pktFiltered);
                }
                ck(av_bsf_send_packet(bsfc, pkt));
                ck(av_bsf_receive_packet(bsfc, pktFiltered));
                *ppData = pktFiltered->data;
                *pnBytes = pktFiltered->size;
                if (pts)
                    *pts = pktFiltered->pts;
                if (dts)
                    *dts = pktFiltered->dts;
            }
            else {

                if (bMp4MPEG4 && (frameCount == 0)) {

                    int extraDataSize = fmtc->streams[iVideoStream]->codecpar->extradata_size;

                    if (extraDataSize > 0) {

                        // extradata contains start codes 00 00 01. Subtract its size
                        pDataWithHeader = (uint8_t*)av_malloc(extraDataSize + pkt->size - 3 * sizeof(uint8_t));

                        if (!pDataWithHeader) {
                            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
                            return false;
                        }

                        memcpy(pDataWithHeader, fmtc->streams[iVideoStream]->codecpar->extradata, extraDataSize);
                        memcpy(pDataWithHeader + extraDataSize, pkt->data + 3, pkt->size - 3 * sizeof(uint8_t));

                        *ppData = pDataWithHeader;
                        *pnBytes = extraDataSize + pkt->size - 3 * sizeof(uint8_t);
                    }

                }
                else {
                    *ppData = pkt->data;
                    *pnBytes = pkt->size;
                }

                if (pts)
                    *pts = pkt->pts;
                if (dts)
                    *dts = pkt->dts;
            }

            if (isVideoPacket)
                *isVideoPacket = 1;

            frameCount++;
        }
        else
        {
            *ppData = pkt->data;
            *pnBytes = pkt->size;
            if (pts)
                *pts = pkt->pts;
            if (dts)
                *dts = pkt->dts;
            if (isVideoPacket)
                *isVideoPacket = 0;
        }

        if (streamIndex)
            *streamIndex = pkt->stream_index;

        return true;
    }

    static int ReadPacket(void *opaque, uint8_t *pBuf, int nBuf) {
        return ((DataProvider *)opaque)->GetData(pBuf, nBuf);
    }
};

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO : return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO : return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4      : return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3       :
    case AV_CODEC_ID_VC1        : return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264       : return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC       : return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8        : return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9        : return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG      : return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_AV1        : return cudaVideoCodec_AV1;
    default                     : return cudaVideoCodec_NumCodecs;
    }
}


