import unittest
import fwt
import numpy
import math
import random


def ifwt1d_ref(decomp_src, decomp_wav, scaling_coef, wavelet_coef):
    """ 1次元高速ウェーブレット逆変換（リファレンス） """
    half_coef_len = len(scaling_coef) // 2
    decomp_src_len = len(decomp_src)
    src = numpy.zeros(2 * decomp_src_len)
    for i in range(decomp_src_len):
        for j in range(half_coef_len):
            index = (i - j) % decomp_src_len
            decomp_s = decomp_src[index]
            decomp_w = decomp_wav[index]
            src[2 * i] += \
                    scaling_coef[2 * j] * decomp_s + wavelet_coef[2 * j] * decomp_w
            src[2 * i + 1] += \
                    scaling_coef[2 * j + 1] * decomp_s + wavelet_coef[2 * j + 1] * decomp_w
    return src


def fwt1d_ref(src, scaling_coef, wavelet_coef):
    """ 1次元高速ウェーブレット変換（リファレンス） """
    src_len = len(src)
    half_len = src_len // 2
    coef_len = len(scaling_coef)
    decomp_src = numpy.zeros(half_len)
    decomp_wav = numpy.zeros(half_len)
    for i in range(half_len):
        for j in range(coef_len):
            index = (j + 2 * i) % src_len
            decomp_src[i] += scaling_coef[j] * src[index]
            decomp_wav[i] += wavelet_coef[j] * src[index]
    return [decomp_src, decomp_wav]


class TestFWT(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_generate_scaling_coef(self):
        """ スケーリング係数の生成テスト """
        haar_wav_coef = numpy.array([1, 1])
        haar_scaling_coef = numpy.array([1, -1])
        haar_wav_coef = haar_wav_coef / numpy.linalg.norm(haar_wav_coef)
        haar_scaling_coef = haar_scaling_coef / numpy.linalg.norm(haar_scaling_coef)
        test_coef = fwt.generate_scaling_coef(haar_wav_coef)
        self.assertEqual(haar_scaling_coef.tolist(), test_coef.tolist())

    def test_ref_decomp_comp_by_haar(self):
        """ リファレンス実装・ハール基底による分解・再合成テスト """
        haar_wav_coef = numpy.array([1, 1])
        haar_wav_coef = haar_wav_coef / numpy.linalg.norm(haar_wav_coef)
        haar_scaling_coef = fwt.generate_scaling_coef(haar_wav_coef)
        # 無音
        src = numpy.zeros(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # 直流
        src = numpy.ones(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # -1,1繰り返し振動
        src = numpy.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # 長めの正弦波
        src = numpy.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # ホワイトノイズ
        src = numpy.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())

    def test_ref_decomp_comp_by_daubechies(self):
        """ リファレンス実装・ドベシィ基底による分解・再合成テスト """
        wav_coef = numpy.array([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551])
        wav_coef = wav_coef / numpy.linalg.norm(wav_coef)
        scaling_coef = fwt.generate_scaling_coef(wav_coef)
        # 無音
        src = numpy.zeros(8)
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef, wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # 直流
        src = numpy.ones(8)
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef, wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # -1,1繰り返し振動
        src = numpy.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef, wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # 長めの正弦波
        src = numpy.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef, wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())
        # ホワイトノイズ
        src = numpy.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef, wav_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, comp_src).all())

    def test_decomp_ref_by_haar(self):
        """ 分解結果の一致確認テスト """
        haar_wav_coef = numpy.array([1, 1])
        haar_wav_coef = haar_wav_coef / numpy.linalg.norm(haar_wav_coef)
        haar_scaling_coef = fwt.generate_scaling_coef(haar_wav_coef)
        # 無音
        src = numpy.zeros(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(numpy.isclose(decomp_wav, decomp_wav_test).all())
        # 直流
        src = numpy.ones(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(numpy.isclose(decomp_wav, decomp_wav_test).all())
        # -1,1繰り返し振動
        src = numpy.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(numpy.isclose(decomp_wav, decomp_wav_test).all())
        # 長めの正弦波
        src = numpy.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(numpy.isclose(decomp_wav, decomp_wav_test).all())
        # ホワイトノイズ
        src = numpy.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef, haar_wav_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(numpy.isclose(decomp_wav, decomp_wav_test).all())

    def test_decomp_comp_by_haar(self):
        """ ハール基底による分解・再合成テスト """
        haar_wav_coef = numpy.array([1, 1])
        haar_wav_coef = haar_wav_coef / numpy.linalg.norm(haar_wav_coef)
        haar_scaling_coef = fwt.generate_scaling_coef(haar_wav_coef)
        # 無音
        src = numpy.zeros(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # 直流
        src = numpy.ones(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # -1,1繰り返し振動
        src = numpy.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # 長めの正弦波
        src = numpy.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # ホワイトノイズ
        src = numpy.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())

    def test_decomp_comp_by_daubechies(self):
        """ ドベシィ基底による分解・再合成テスト """
        wav_coef = numpy.array([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551])
        wav_coef = wav_coef / numpy.linalg.norm(wav_coef)
        scaling_coef = fwt.generate_scaling_coef(wav_coef)
        # 無音
        src = numpy.zeros(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef, wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # 直流
        src = numpy.ones(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef, wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # -1,1繰り返し振動
        src = numpy.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef, wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # 長めの正弦波
        src = numpy.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef, wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # ホワイトノイズ
        src = numpy.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef, wav_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef, wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())

    def test_2d_decomp_comp_by_haar(self):
        """ ハール基底による2次元分解・再合成テスト """
        haar_wav_coef = numpy.array([-1, 1])
        haar_wav_coef = haar_wav_coef / numpy.linalg.norm(haar_wav_coef)
        haar_scaling_coef = fwt.generate_scaling_coef(haar_wav_coef)
        # 全て0
        src = numpy.zeros((4,4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # 全て1
        src = numpy.ones((4,4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # -1,1繰り返し振動
        src = numpy.array([(-1)**(i) for i in range(4 * 4)]).reshape((4,4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # 長めの正弦波
        src = numpy.array([math.sin(i) for i in range(16 * 16)]).reshape((16, 16))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())
        # ホワイトノイズ
        src = numpy.array([random.random() for i in range(4 * 4)]).reshape((4, 4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef, haar_wav_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef, haar_wav_coef)
        self.assertTrue(numpy.isclose(src, src_test).all())

if __name__ == '__main__':
    unittest.main()
